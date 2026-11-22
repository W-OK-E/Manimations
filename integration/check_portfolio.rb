# Focused check against the real portfolio Liquid header and new integration files.
# This does not substitute for the complete Jekyll/Docker production build.
require 'yaml'
require 'liquid'
require 'kramdown'
require 'pathname'

ROOT = Pathname.new(ARGV.fetch(0)).expand_path
OUTPUT = Pathname.new(ARGV.fetch(1, 'build/integration/portfolio-preview')).expand_path

def document(path)
  content = path.read
  match = content.match(/\A---\s*\n(.*?)\n---\s*\n(.*)\z/m)
  raise "Missing front matter: #{path}" unless match
  [YAML.safe_load(match[1]), match[2]]
end

module Urls
  def relative_url(input)
    return input if input.to_s.match?(%r{\Ahttps?://})
    base = @context.registers[:baseurl].to_s.sub(%r{/$}, '')
    "#{base}/#{input.to_s.sub(%r{^/}, '')}"
  end
end

class IncludeFile < Liquid::Tag
  def initialize(name, markup, tokens)
    super
    @name = markup.strip
  end
  def render(context)
    Liquid::Template.parse((ROOT / '_includes' / @name).read).render!(context)
  end
end

Liquid::Template.register_filter(Urls)
Liquid::Template.register_tag('include', IncludeFile)
page, body = document(ROOT / '_pages/animations.md')
project, project_body = document(ROOT / '_projects/12_manimations.md')
data = YAML.safe_load((ROOT / '_data/manimations.yml').read)
raise 'gallery_path must be blank or /Manimations/' unless ['', '/Manimations/'].include?(data['gallery_path'])
raise 'nav must be a YAML boolean' unless [true, false].include?(page['nav'])
raise 'Gallery button points back to its own landing page' if data['gallery_path'] == page['permalink']
header = Liquid::Template.parse((ROOT / '_includes/header.liquid').read)
template = Liquid::Template.parse(body)
OUTPUT.mkpath

def site_context(page, data, baseurl)
  {'page' => page.merge('url' => page['permalink']),
   'site' => {'title' => 'Om Kumar', 'baseurl' => baseurl,
              'pages' => [{'permalink' => '/', 'title' => 'about', 'url' => '/'}, page.merge('url' => page['permalink'])],
              'data' => {'manimations' => data}}}
end

checks = 0
[false, true].each do |visible|
  ['', '/preview'].each do |baseurl|
    assigned_page = page.merge('nav' => visible)
    context = site_context(assigned_page, {'gallery_path' => '/Manimations/'}, baseurl)
    rendered_header = header.render!(context, registers: {baseurl: baseurl})
    nav_link = %(href="#{baseurl}#{page['permalink']}")
    raise "Navbar toggle incorrect for #{visible}" unless rendered_header.include?(nav_link) == visible
    rendered_body = Kramdown::Document.new(template.render!(context, registers: {baseurl: baseurl})).to_html
    raise 'Gallery path incorrect' unless rendered_body.include?(%(href="#{baseurl}/Manimations/"))
    project_html = Kramdown::Document.new(Liquid::Template.parse(project_body).render!(context, registers: {baseurl: baseurl})).to_html
    raise 'Project loses gallery link when navbar hidden' unless project_html.include?(%(href="#{baseurl}/Manimations/"))
    raise 'Project About link does not match current landing permalink' unless project_html.include?(%(href="#{baseurl}#{page['permalink']}"))
    (OUTPUT / "nav-#{visible}-#{baseurl.empty? ? 'root' : 'subpath'}.html").write("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Portfolio integration preview</title>#{rendered_header}<main>#{rendered_body}</main></html>")
    checks += 1
  end
end
blank_context = site_context(page, {'gallery_path' => ''}, '')
blank = template.render!(blank_context, registers: {baseurl: ''})
raise 'Blank TODO must not produce a broken gallery link' if blank.include?('Open the animation gallery')
raise 'Missing pending text' unless blank.include?('coming soon')
puts "PASS: #{checks} navbar/baseurl combinations, project link independent of navbar, safe blank URL."
puts "Rendered focused Liquid/Markdown fixtures: #{OUTPUT}. Full Jekyll/Docker build remains separate."
