// Pure API mocks: this file never contacts GitHub or publishes anything.
const assert = require('node:assert/strict');
const deploy = require('../.github/scripts/deploy-pages.cjs');

function fixture({ artifactChanges = {}, statuses = ['queued', 'succeed'], metadata404s = 0, pollErrors = [] } = {}) {
  const calls = [], logs = [], outputs = {}, sleeps = [];
  let missing = metadata404s;
  const args = {
    context: { repo: { owner: 'W-OK-E', repo: 'Manimations' }, runId: 123, sha: 'commit-sha' },
    artifactId: '456', maxPolls: 3,
    core: { info: value => logs.push(value), warning: value => logs.push(value),
            getIDToken: async () => 'SECRET-MOCK-OIDC', setOutput: (key, value) => { outputs[key] = value; } },
    sleep: async value => sleeps.push(value),
    github: { request: async (route, parameters) => {
      calls.push({ route, parameters });
      if (route.includes('/actions/artifacts/')) {
        if (missing-- > 0) throw Object.assign(new Error('not yet visible'), { status: 404 });
        return { data: { id: 456, name: 'github-pages', size_in_bytes: 1000, expired: false,
                         workflow_run: { id: 123 }, ...artifactChanges } };
      }
      if (route === 'POST /repos/{owner}/{repo}/pages/deployments') return { data: { id: 'deployment-sha', page_url: 'https://w-ok-e.github.io/Manimations/' } };
      if (route.endsWith('/cancel')) return { status: 204 };
      if (route.startsWith('GET /repos/{owner}/{repo}/pages/deployments/')) {
        if (pollErrors.length) throw Object.assign(new Error('temporary failure'), { status: pollErrors.shift() });
        return { data: { status: statuses.shift() || 'queued' } };
      }
      throw new Error(`Unexpected route: ${route}`);
    } },
  };
  return { args, calls, logs, outputs, sleeps };
}

(async () => {
  const successful = fixture({ metadata404s: 1 });
  assert.equal(await deploy(successful.args), 'https://w-ok-e.github.io/Manimations/');
  const creation = successful.calls.find(call => call.route === 'POST /repos/{owner}/{repo}/pages/deployments');
  assert.equal(creation.parameters.artifact_id, 456);
  assert.equal(creation.parameters.oidc_token, 'SECRET-MOCK-OIDC');
  assert.equal(creation.parameters.pages_build_version, 'commit-sha');
  assert.equal(successful.outputs.page_url, 'https://w-ok-e.github.io/Manimations/');
  assert.deepEqual(successful.sleeps, [1000, 5000]);
  assert(!JSON.stringify(successful.logs).includes('SECRET-MOCK-OIDC'));
  assert(successful.calls.every(call => !call.route.endsWith('/artifacts')), 'No name-list/discovery endpoint');
  const intermittent = fixture({ pollErrors: [404, 503], statuses: ['succeed'] });
  await deploy(intermittent.args);
  assert.equal(intermittent.calls.filter(call => call.route === 'POST /repos/{owner}/{repo}/pages/deployments').length, 1);
  assert.deepEqual(intermittent.sleeps, [5000, 5000]);

  for (const artifactChanges of [{ name: 'wrong' }, { expired: true }, { workflow_run: { id: 999 } }]) {
    const wrong = fixture({ artifactChanges });
    await assert.rejects(deploy(wrong.args), /artifact must be/);
    assert(!wrong.calls.some(call => call.route.startsWith('POST')));
  }
  const badId = fixture();
  badId.args.artifactId = '';
  await assert.rejects(deploy(badId.args), /valid Pages artifact ID/);
  assert.equal(badId.calls.length, 0);

  const failed = fixture({ statuses: ['deployment_failed'] });
  await assert.rejects(deploy(failed.args), /Pages deployment failed/);
  const timeout = fixture({ statuses: [] });
  await assert.rejects(deploy(timeout.args), /Timed out/);
  assert(timeout.calls.some(call => call.route.endsWith('/cancel')));
  assert.equal(timeout.calls.filter(call => call.route === 'POST /repos/{owner}/{repo}/pages/deployments').length, 1);

  const forbidden = fixture();
  forbidden.args.github.request = async () => { throw Object.assign(new Error('Forbidden'), { status: 403 }); };
  await assert.rejects(deploy(forbidden.args), /Forbidden/);
  assert.equal(forbidden.sleeps.length, 0);
  console.log('PASS: exact uploaded ID, same-run verification, bounded visibility retry, OIDC payload without token logs, success/failure/timeout, invalid artifact rejection, no name discovery. All API calls mocked.');
})().catch(error => { console.error(error); process.exit(1); });
