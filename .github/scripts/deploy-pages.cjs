// Use the uploader's exact ID, avoiding deploy-pages' failing artifact-name lookup.
// API contract: https://docs.github.com/en/rest/pages/pages#create-a-github-pages-deployment
module.exports = async function deployPages({
  github, core, context, artifactId,
  sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds)),
  maxPolls = 60,
}) {
  if (!/^[1-9]\d*$/.test(String(artifactId)) || !Number.isSafeInteger(Number(artifactId))) {
    throw new Error('The build job did not return a valid Pages artifact ID. Check its upload step/output.');
  }
  const id = Number(artifactId);
  const repository = { owner: context.repo.owner, repo: context.repo.repo };
  let artifact;
  for (let attempt = 1; attempt <= 6; attempt++) {
    try {
      ({ data: artifact } = await github.request('GET /repos/{owner}/{repo}/actions/artifacts/{artifact_id}', {
        ...repository, artifact_id: id,
      }));
      break;
    } catch (error) {
      if (error.status !== 404 || attempt === 6) throw error;
      core.info(`Waiting for uploaded artifact ${id} to become visible (${attempt}/6).`);
      await sleep(attempt * 1000);
    }
  }
  if (artifact.name !== 'github-pages' || artifact.expired || Number(artifact.workflow_run?.id) !== Number(context.runId)) {
    throw new Error('The uploaded artifact must be named github-pages, unexpired, and belong to this workflow run.');
  }
  core.info(`Verified github-pages artifact ${id}: ${artifact.size_in_bytes} bytes, run ${context.runId}.`);

  // Never log the OIDC token or the request payload containing it.
  const oidcToken = await core.getIDToken();
  const { data: created } = await github.request('POST /repos/{owner}/{repo}/pages/deployments', {
    ...repository, artifact_id: id, pages_build_version: context.sha,
    oidc_token: oidcToken, environment: 'github-pages',
  });
  if (!created.id) throw new Error('GitHub did not return a Pages deployment ID. Inspect the deployment in GitHub before retrying.');
  core.setOutput('page_url', created.page_url);
  core.info(`Pages deployment created: ${created.id}.`);
  const failed = new Set(['deployment_failed', 'deployment_content_failed', 'deployment_cancelled', 'deployment_lost', 'failed']);
  for (let attempt = 0; attempt < maxPolls; attempt++) {
    let data;
    try {
      ({ data } = await github.request('GET /repos/{owner}/{repo}/pages/deployments/{pages_deployment_id}', {
        ...repository, pages_deployment_id: created.id,
      }));
    } catch (error) {
      if (error.status !== 404 && error.status !== 429 && !(error.status >= 500)) throw error;
      core.info(`Pages status temporarily unavailable (HTTP ${error.status}); polling the existing deployment again.`);
      if (attempt + 1 < maxPolls) await sleep(5000);
      continue;
    }
    core.info(`Pages deployment status: ${data.status}`);
    if (data.status === 'succeed') return created.page_url;
    if (failed.has(data.status)) throw new Error(`Pages deployment failed: ${data.status}`);
    if (attempt + 1 < maxPolls) await sleep(5000);
  }
  try {
    await github.request('POST /repos/{owner}/{repo}/pages/deployments/{pages_deployment_id}/cancel', {
      ...repository, pages_deployment_id: created.id,
    });
  } catch {
    core.warning('Timed-out deployment could not be cancelled; inspect its state in GitHub.');
  }
  throw new Error('Timed out waiting for GitHub Pages to publish.');
};
