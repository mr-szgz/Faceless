This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.editorconfig
.github/dependabot.yml
.github/workflows/ruby.yml
.gitignore
.ruby-version
bin/rake
CHANGELOG.md
CODE_OF_CONDUCT.md
config.rb
CONTRIBUTING.md
data/links.json
Gemfile
LICENSE
Rakefile
README.md
source/ar/1.0.0/index.html.haml
source/assets/images/bg-hero@2x.png
source/assets/images/bg-img1@2x.png
source/assets/images/bg-img2@2x.png
source/assets/images/favicon.ico
source/assets/images/keep-a-changelog-mark.svg
source/assets/images/keep-a-changelog-opengraph.png
source/assets/images/logo.png
source/assets/images/logo.svg
source/assets/javascripts/all.js
source/assets/stylesheets/_mixins.sass
source/assets/stylesheets/_variables.sass
source/assets/stylesheets/anchors.sass
source/assets/stylesheets/application.css.sass
source/assets/stylesheets/layout.sass
source/assets/stylesheets/legacy.css.sass
source/assets/stylesheets/sections.sass
source/assets/stylesheets/table-of-contents.sass
source/CNAME
source/cs/0.3.0/index.html.haml
source/cs/1.0.0/index.html.haml
source/cs/1.1.0/index.html.haml
source/da/1.1.0/index.html.haml
source/de/0.3.0/index.html.haml
source/de/1.0.0/index.html.haml
source/de/1.1.0/index.html.haml
source/en/0.3.0/index.html.haml
source/en/1.0.0/index.html.haml
source/en/1.1.0/index.html.haml
source/es-ES/0.3.0/index.html.haml
source/es-ES/1.0.0/index.html.haml
source/es-ES/1.1.0/index.html.haml
source/fa/1.0.0/index.html.haml
source/fa/1.1.0/index.html.haml
source/fr/0.3.0/index.html.haml
source/fr/1.0.0/index.html.haml
source/fr/1.1.0/index.html.haml
source/hr/1.0.0/index.html.haml
source/id-ID/1.0.0/index.html.haml
source/id/1.1.0/index.html.haml
source/it-IT/0.3.0/index.html.haml
source/it-IT/1.0.0/index.html.haml
source/it-IT/1.1.0/index.html.haml
source/ja/1.0.0/index.html.haml
source/ja/1.1.0/index.html.haml
source/ka/1.0.0/index.html.haml
source/ko/1.0.0/index.html.haml
source/ko/1.1.0/index.html.haml
source/layouts/layout.html.haml
source/nb/1.1.0/index.html.haml
source/nl/1.0.0/index.html.haml
source/nl/1.1.0/index.html.haml
source/pl/0.3.0/index.html.haml
source/pl/1.0.0/index.html.haml
source/pl/1.1.0/index.html.haml
source/pt-BR/0.3.0/index.html.haml
source/pt-BR/1.0.0/index.html.haml
source/pt-BR/1.1.0/index.html.haml
source/ro/1.1.0/index.html.haml
source/ru/0.3.0/index.html.haml
source/ru/1.0.0/index.html.haml
source/ru/1.1.0/index.html.haml
source/sk/1.0.0/index.html.haml
source/sl/0.3.0/index.html.haml
source/sl/1.0.0/index.html.haml
source/sl/1.1.0/index.html.haml
source/sr/1.0.0/index.html.haml
source/sv/0.3.0/index.html.haml
source/sv/1.0.0/index.html.haml
source/sv/1.1.0/index.html.haml
source/tr-TR/0.3.0/index.html.haml
source/tr-TR/1.0.0/index.html.haml
source/tr-TR/1.1.0/index.html.haml
source/uk/1.0.0/index.html.haml
source/uk/1.1.0/index.html.haml
source/zh-CN/0.3.0/index.html.haml
source/zh-CN/1.0.0/index.html.haml
source/zh-CN/1.1.0/index.html.haml
source/zh-TW/0.3.0/index.html.haml
source/zh-TW/1.0.0/index.html.haml
source/zh-TW/1.1.0/index.html.haml
test/build_test.rb
```

# Files

## File: .editorconfig
```
# EditorConfig is awesome: http://EditorConfig.org
#
# All EditorConfig properties:
# https://github.com/editorconfig/editorconfig/wiki/EditorConfig-Properties


# Top-most EditorConfig file
root = true

# This is a small project; currently, we
# use the same settings everywhere
[*]
end_of_line = lf
charset = utf-8
insert_final_newline = true
indent_style = space
indent_size = 2
```

## File: .github/dependabot.yml
```yaml
version: 2
updates:
- package-ecosystem: bundler
  directory: "/"
  schedule:
    interval: monthly
    time: "04:00"
  open-pull-requests-limit: 3
  ignore:
  - dependency-name: middleman
    versions:
    - 4.3.7
```

## File: .github/workflows/ruby.yml
```yaml
# This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.
# This workflow will download a prebuilt Ruby version, install dependencies and run tests with Rake
# For more information see: https://github.com/marketplace/actions/setup-ruby-jruby-and-truffleruby

name: Ruby

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

permissions:
  contents: read

jobs:
  test:

    runs-on: ubuntu-latest
    strategy:
      matrix:
        ruby-version: ['3.3']

    steps:
    - uses: actions/checkout@v4
    - name: Set up Ruby
    # To automatically get bug fixes and new Ruby versions for ruby/setup-ruby,
    # change this to (see https://github.com/ruby/setup-ruby#versioning):
    # uses: ruby/setup-ruby@v1
      uses: ruby/setup-ruby@master
      with:
        ruby-version: ${{ matrix.ruby-version }}
        bundler-cache: true # runs 'bundle install' and caches installed gems automatically
    - name: Run tests
      run: bin/rake test
```

## File: .gitignore
```
.DS_Store
/.bundle
/.cache
/.sass-cache
/build
/_site
/.vs
/.idea

*~
\#*#
```

## File: .ruby-version
```
3.3.1
```

## File: bin/rake
```
#!/usr/bin/env ruby
# frozen_string_literal: true

#
# This file was generated by Bundler.
#
# The application 'rake' is installed as part of a gem, and
# this file is here to facilitate running it.
#

require "pathname"
ENV["BUNDLE_GEMFILE"] ||= File.expand_path("../../Gemfile",
  Pathname.new(__FILE__).realpath)

bundle_binstub = File.expand_path("../bundle", __FILE__)

if File.file?(bundle_binstub)
  if File.read(bundle_binstub, 300) =~ /This file was generated by Bundler/
    load(bundle_binstub)
  else
    abort("Your `bin/bundle` was not generated by Bundler, so this binstub cannot run.
Replace `bin/bundle` by running `bundle binstubs bundler --force`, then run this command again.")
  end
end

require "rubygems"
require "bundler/setup"

load Gem.bin_path("rake", "rake")
```

## File: CHANGELOG.md
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- v1.1 Brazilian Portuguese translation.
- v1.1 German Translation
- v1.1 Spanish translation.
- v1.1 Italian translation.
- v1.1 Polish translation.
- v1.1 Ukrainian translation.
- v1.1 Swedish translation (#590).

### Changed

- Use frontmatter title & description in each language version template
- Replace broken OpenGraph image with an appropriately-sized Keep a Changelog 
  image that will render properly (although in English for all languages)
- Fix OpenGraph title & description for all languages so the title and 
description when links are shared are language-appropriate

### Removed

- Trademark sign previously shown after the project description in version 
0.3.0

## [1.1.1] - 2023-03-05

### Added

- v1.1 Arabic translation (#444).
- v1.1 French translation.
- v1.1 Dutch translation (#371).
- v1.1 Russian translation (#410).
- v1.1 Japanese translation (#363).
- v1.1 Norwegian Bokmål translation (#383).
- v1.1 "Inconsistent Changes" Turkish translation (#347).
- Default to most recent versions available for each languages.
- Display count of available translations (26 to date!).
- Centralize all links into `/data/links.json` so they can be updated easily.

### Fixed

- Improve French translation (#377).
- Improve id-ID translation (#416).
- Improve Persian translation (#457).
- Improve Russian translation (#408).
- Improve Swedish title (#419).
- Improve zh-CN translation (#359).
- Improve French translation (#357).
- Improve zh-TW translation (#360, #355).
- Improve Spanish (es-ES) transltion (#362).
- Foldout menu in Dutch translation (#371).
- Missing periods at the end of each change (#451).
- Fix missing logo in 1.1 pages.
- Display notice when translation isn't for most recent version.
- Various broken links, page versions, and indentations.

### Changed

- Upgrade dependencies: Ruby 3.2.1, Middleman, etc.

### Removed

- Unused normalize.css file.
- Identical links assigned in each translation file.
- Duplicate index file for the english version.

## [1.1.0] - 2019-02-15

### Added

- Danish translation (#297).
- Georgian translation from (#337).
- Changelog inconsistency section in Bad Practices.

### Fixed

- Italian translation (#332).
- Indonesian translation (#336).

## [1.0.0] - 2017-06-20

### Added

- New visual identity by [@tylerfortune8](https://github.com/tylerfortune8).
- Version navigation.
- Links to latest released version in previous versions.
- "Why keep a changelog?" section.
- "Who needs a changelog?" section.
- "How do I make a changelog?" section.
- "Frequently Asked Questions" section.
- New "Guiding Principles" sub-section to "How do I make a changelog?".
- Simplified and Traditional Chinese translations from [@tianshuo](https://github.com/tianshuo).
- German translation from [@mpbzh](https://github.com/mpbzh) & [@Art4](https://github.com/Art4).
- Italian translation from [@azkidenz](https://github.com/azkidenz).
- Swedish translation from [@magol](https://github.com/magol).
- Turkish translation from [@emreerkan](https://github.com/emreerkan).
- French translation from [@zapashcanon](https://github.com/zapashcanon).
- Brazilian Portuguese translation from [@Webysther](https://github.com/Webysther).
- Polish translation from [@amielucha](https://github.com/amielucha) & [@m-aciek](https://github.com/m-aciek).
- Russian translation from [@aishek](https://github.com/aishek).
- Czech translation from [@h4vry](https://github.com/h4vry).
- Slovak translation from [@jkostolansky](https://github.com/jkostolansky).
- Korean translation from [@pierceh89](https://github.com/pierceh89).
- Croatian translation from [@porx](https://github.com/porx).
- Persian translation from [@Hameds](https://github.com/Hameds).
- Ukrainian translation from [@osadchyi-s](https://github.com/osadchyi-s).

### Changed

- Start using "changelog" over "change log" since it's the common usage.
- Start versioning based on the current English version at 0.3.0 to help
  translation authors keep things up-to-date.
- Rewrite "What makes unicorns cry?" section.
- Rewrite "Ignoring Deprecations" sub-section to clarify the ideal
  scenario.
- Improve "Commit log diffs" sub-section to further argument against
  them.
- Merge "Why can’t people just use a git log diff?" with "Commit log
  diffs".
- Fix typos in Simplified Chinese and Traditional Chinese translations.
- Fix typos in Brazilian Portuguese translation.
- Fix typos in Turkish translation.
- Fix typos in Czech translation.
- Fix typos in Swedish translation.
- Improve phrasing in French translation.
- Fix phrasing and spelling in German translation.

### Removed

- Section about "changelog" vs "CHANGELOG".

## [0.3.0] - 2015-12-03

### Added

- RU translation from [@aishek](https://github.com/aishek).
- pt-BR translation from [@tallesl](https://github.com/tallesl).
- es-ES translation from [@ZeliosAriex](https://github.com/ZeliosAriex).

## [0.2.0] - 2015-10-06

### Changed

- Remove exclusionary mentions of "open source" since this project can
  benefit both "open" and "closed" source projects equally.

## [0.1.0] - 2015-10-06

### Added

- Answer "Should you ever rewrite a change log?".

### Changed

- Improve argument against commit logs.
- Start following [SemVer](https://semver.org) properly.

## [0.0.8] - 2015-02-17

### Changed

- Update year to match in every README example.
- Reluctantly stop making fun of Brits only, since most of the world
  writes dates in a strange way.

### Fixed

- Fix typos in recent README changes.
- Update outdated unreleased diff link.

## [0.0.7] - 2015-02-16

### Added

- Link, and make it obvious that date format is ISO 8601.

### Changed

- Clarified the section on "Is there a standard change log format?".

### Fixed

- Fix Markdown links to tag comparison URL with footnote-style links.

## [0.0.6] - 2014-12-12

### Added

- README section on "yanked" releases.

## [0.0.5] - 2014-08-09

### Added

- Markdown links to version tags on release headings.
- Unreleased section to gather unreleased changes and encourage note
  keeping prior to releases.

## [0.0.4] - 2014-08-09

### Added

- Better explanation of the difference between the file ("CHANGELOG")
  and its function "the change log".

### Changed

- Refer to a "change log" instead of a "CHANGELOG" throughout the site
  to differentiate between the file and the purpose of the file — the
  logging of changes.

### Removed

- Remove empty sections from CHANGELOG, they occupy too much space and
  create too much noise in the file. People will have to assume that the
  missing sections were intentionally left out because they contained no
  notable changes.

## [0.0.3] - 2014-08-09

### Added

- "Why should I care?" section mentioning The Changelog podcast.

## [0.0.2] - 2014-07-10

### Added

- Explanation of the recommended reverse chronological release ordering.

## [0.0.1] - 2014-05-31

### Added

- This CHANGELOG file to hopefully serve as an evolving example of a
  standardized open source project CHANGELOG.
- CNAME file to enable GitHub Pages custom domain.
- README now contains answers to common questions about CHANGELOGs.
- Good examples and basic guidelines, including proper date formatting.
- Counter-examples: "What makes unicorns cry?".

[unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.8...v0.1.0
[0.0.8]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/olivierlacan/keep-a-changelog/releases/tag/v0.0.1
```

## File: CODE_OF_CONDUCT.md
```markdown
# Contributor Covenant Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

## Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at hi@olivierlacan.com. The project team will review and investigate all complaints, and will respond in a way that it deems appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4, available at [https://contributor-covenant.org/version/1/4][version]

[homepage]: https://contributor-covenant.org
[version]: https://contributor-covenant.org/version/1/4/
```

## File: config.rb
```ruby
# --------------------------------------
#   Config
# --------------------------------------

# ----- Site ----- #
# Last version should be the latest English version since the manifesto is first
# written in English, then translated into other languages later.
$versions = Dir.glob("source/en/*").map { |e| e.sub("source/en/", "") }.sort
# $last_version = $versions.last
$last_version = "1.1.0"
$previous_version = $versions[$versions.index($last_version) - 1]

# This list of languages populates the language navigation.
issues_url = "https://github.com/olivierlacan/keep-a-changelog/issues"
$languages = {
  "ar" => {
    name: "العربية"
  },
  "cs" => {
    name: "Čeština"
  },
  "da" => {
    name: "Dansk",
    new: "En ny version er tilgængelig"
  },
  "de" => {
    name: "Deutsch",
    notice: "Die neuste version (#{$last_version}) ist noch nicht auf Deutsch
    verfügbar, aber du kannst sie dir <a href='/en/'>auf Englisch durchlesen</a>
    und <a href='#{issues_url}'>bei der Übersetzung mithelfen</a>."
  },
  "en" => {
    default: true,
    name: "English",
    new: "A new version is available"
  },
  "es-ES" => {
    name: "Español",
    notice: "La última versión (#{$last_version}) aun no está disponible en
    Español, por ahora puedes <a href='/en/'>leerla en Inglés</a> y
    <a href='#{issues_url}'>ayudar a traducirla</a>."
  },
  "fr" => {
    name: "Français",
    notice: "La dernière version (#{$last_version}) n'est pas encore disponible
    en français, mais vous pouvez la <a href='/en/'>lire en anglais</a> pour
    l'instant et <a href='#{issues_url}'>aider à la traduire</a>.",
    new: "Une nouvelle version est disponible"
  },
  "hr" => {
    name: "Hrvatski"
  },
  "id-ID" => {
    name: "Indonesia",
    new: "Ada versi baru tersedia"
  },
  "it-IT" => {
    name: "Italiano",
    notice: "L'ultima versione (#{$last_version}) non è ancora disponibile in
    Italiano, ma la potete <a href='/en/'>leggere in Inglese</a> per ora e
    potete <a href='#{issues_url}'>contribuire a tradurla</a>."
  },
  "ja" => {
    name: "日本語"
  },
  "nb" => {
    name: "Norsk (Bokmål)",
    notice: "Den siste versjonen (#{$last_version}) er ikke tilgjengelig på norsk,
    men du kan <a href='/en/'>lese den på engelsk</a> og <a
    href='#{issues_url}'>hjelpe med å oversette den</a>.",
    new: "En ny versjon er tilgjengelig"
  },
  "nl" => {
    name: "Nederlands"
  },
  "pl" => {
    name: "polski"
  },
  "pt-BR" => {
    name: "Português do Brasil",
    notice: "A última versão (#{$last_version}) ainda não está disponível em
    Português mas nesse momento você pode <a href='/en/'>lê-la em inglês</a> e
    <a href='#{issues_url}'>ajudar em sua tradução</a>."
  },
  "ro" => {
    name: "română",
    new: "O nouă versiune este disponibilă"
  },
  "ru" => {
    name: "Pyccкий",
    notice: "Самая последняя версия (#{$last_version}) ещё пока не переведена на
    русский, но вы можете <a href='/en/'>прочитать её на английском</a> и <a
    href='#{issues_url}'>помочь с переводом</a>."
  },
  "sk" => {
    name: "Slovenčina"
  },
  "ka" => {
    name: "ქართული"
  },
  "sl" => {
    name: "Slovenščina"
  },
  "sr" => {
    name: "Srpski"
  },
  "sv" => {
    name: "Svenska",
    notice: "Den senaste versionen (#{$last_version}) är ännu inte tillgänglig på svenska,
    men du kan <a href='/en/'>läsa det på engelska</a> och även <a
    href='#{issues_url}'>hjälpa till att översätta det</a>."
  },
  "tr-TR" => {
    name: "Türkçe"
  },
  "uk" => {
    name: "Українська"
  },
  "zh-CN" => {
    name: "简体中文",
    notice: "最新版 (#{$last_version}) 暂时还没有翻译到简体中文，您可以阅读最新的英语版，并且帮助翻译，不胜感激。"
  },
  "zh-TW" => {
    name: "正體中文",
    notice: "最新版 (#{$last_version}) 暫時還沒有翻譯到正體中文，您可以閱讀最新的英語版，並且幫助翻譯，不勝感激。"
  },
  "ko" => {
    name: "한국어"
  },
  "fa" => {
    name: "فارسی"
  }
}
$language_count = $languages.size

activate :i18n,
  lang_map: $languages,
  mount_at_root: :en

set :gauges_id, ""
set :publisher_url, "https://www.facebook.com/olivier.lacan.5"
set :site_url, "https://keepachangelog.com"

redirect "index.html", to: "en/#{$last_version}/index.html"

$languages.each do |language|
  code = language.first
  versions = Dir.entries("source/#{code}").sort - %w[. ..]
  redirect "#{code}/index.html", to: "#{code}/#{versions.last}/index.html"
end

# ----- Assets ----- #

set :css_dir, "assets/stylesheets"
set :js_dir, "assets/javascripts"
set :images_dir, "assets/images"
set :fonts_dir, "assets/fonts"

# ----- Images ----- #

activate :automatic_image_sizes

# ----- Markdown ----- #

activate :syntax
set :markdown_engine, :redcarpet

## Override default Redcarpet renderer in order to define a class
class CustomMarkdownRenderer < Redcarpet::Render::HTML
  def doc_header
    %(<nav class="toc">#{@header}</nav>)
  end

  def header(text, header_level)
    slug = text.parameterize
    tag_name = "h#{header_level}"
    anchor_link = "<a id='#{slug}' class='anchor' href='##{slug}' aria-hidden='true'></a>"
    header_tag_open = "<#{tag_name} id='#{slug}'>"

    output = ""
    output << header_tag_open
    output << anchor_link
    output << text
    output << "</#{tag_name}>"

    output
  end
end

$markdown_config = {
  fenced_code_blocks: true,
  footnotes: true,
  smartypants: true,
  tables: true,
  with_toc_data: true,
  renderer: CustomMarkdownRenderer
}
set :markdown, $markdown_config

# --------------------------------------
#   Helpers
# --------------------------------------

helpers do
  def path_to_url(path)
    Addressable::URI.join(config.site_url, path).normalize.to_s
  end

  def available_translation_for(language)
    language_name = language.last[:name]
    language_path = "source/#{language.first}"

    if File.exist?("#{language_path}/#{$last_version}")
      "#{$last_version} #{language_name}"
    elsif File.exist?("#{language_path}/#{$previous_version}")
      "#{$previous_version} #{language_name}"
    end
  end
end

# --------------------------------------
#   Content
# --------------------------------------

# ----- Directories ----- #

activate :directory_indexes
page "/404.html", directory_index: false

# --------------------------------------
#   Production
# --------------------------------------

# ----- Optimization ----- #

configure :build do
  set :gauges_id, "5389808eeddd5b055a00440d"
  activate :asset_hash
  activate :gzip, {exts: %w[
    .css
    .eot
    .htm
    .html
    .ico
    .js
    .json
    .svg
    .ttf
    .txt
    .woff
  ]}
  activate :minify_css
  activate :minify_html do |html|
    html.remove_quotes = false
  end
  activate :minify_javascript
end

# ----- Prefixing ----- #

activate :autoprefixer do |config|
  config.browsers = ["last 2 versions", "Explorer >= 10"]
  config.cascade = false
end
```

## File: CONTRIBUTING.md
```markdown
See [README](README.md#development).
```

## File: data/links.json
```json
{
  "changelog": "https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md",
  "repo": "https://github.com/olivierlacan/keep-a-changelog",
  "issue": "https://github.com/olivierlacan/keep-a-changelog/issues",
  "semver": "https://semver.org/",
  "shields": "https://shields.io/",
  "thechangelog": "https://changelog.com/podcast/127",
  "vandamme": "https://github.com/tech-angels/vandamme/",
  "isodate": "https://www.iso.org/iso-8601-date-and-time-format.html",
  "github_releases": "https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository",
  "gnustyle": "https://www.gnu.org/prep/standards/html_node/Style-of-Change-Logs.html#Style-of-Change-Logs",
  "gnunews": "https://www.gnu.org/prep/standards/html_node/NEWS-File.html#NEWS-File"
}
```

## File: Gemfile
```
source "https://rubygems.org"

gem "addressable"
gem "middleman", "~> 4.5.1"
gem "middleman-autoprefixer"
gem "middleman-blog"
gem "middleman-livereload"
gem "middleman-minify-html"
gem "middleman-syntax"
gem "middleman-gh-pages"
gem "redcarpet"
gem "standard", "~> 1.40"

group :development, :test do 
  gem "minitest"
end
```

## File: LICENSE
```
The MIT License (MIT)

Copyright (c) 2014 Olivier Lacan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## File: Rakefile
```
require "middleman-gh-pages"

desc "Preview build on localhost:4567 with live reload"
task :serve do
  puts "Running local development server on http://localhost:4567"
  system("bundle exec middleman serve")
end

desc "Clean, build and publish to GitHub Pages"
task deploy: [:clean, :publish]

desc "Build middleman static site"
task :build do
  puts "Build site into build/ directory, print any errors"
  system("bundle exec middleman build --verbose")
end

desc "Delete build directory"
task :clean do
  puts "Cleaning build/ directory"
  system("rm -rf build/")
end

require "minitest/test_task"

Minitest::TestTask.create(:test) do |t|
  t.libs << "test"
  # t.libs << "lib"
  t.warning = false
  t.test_globs = ["test/*_test.rb"]
end

task :default => :test
```

## File: README.md
```markdown
# <img src="https://d3vv6lp55qjaqc.cloudfront.net/items/1L1w0v431V0d1K410f3Y/keepAChangelog-logo-dark.svg" height=150 alt="Keep a Changelog" />

[![Keep a Changelog v1.1.0 badge][changelog-badge]][changelog] [![Version 1.1.0 Badge][version-badge]][changelog] [![MIT License Badge][license-badge]][license]

Don’t let your friends dump git logs into changelogs™

This repository generates https://keepachangelog.com/.

## Development

### Dependencies

- Ruby ([see version][ruby-version], [rbenv][rbenv] recommended)
- Bundler (`gem install bundler`)

### Installation

- `git clone https://github.com/olivierlacan/keep-a-changelog.git`
- `cd keep-a-changelog`
- `bundle install`
- `bin/rake serve` starts a local development server at http://localhost:4567
  which will reload with any local file changes
- `bin/rake build` runs middleman build with `--verbose` flag so build errors are 
  logged for easier debugging

### Deployment

- `bin/rake clean` can clean a corrupted `build/` directory in 
  case `publish` failed
- `bin/rake deploy` cleans, builds and pushes to the `gh-pages` branch on GitHub so 
  the site is deployed to keepachangelog.com

### Translations

Create a new directory in [`source/`][source] named after the ISO 639-1 code
for the language you wish to translate Keep a Changelog to. For example,
assuming you want to translate to French Canadian:

- create the `source/fr-CA` directory.
- duplicate the `source/en/1.0.0/index.html.haml` file in `source/fr-CA`.
- edit `source/fr-CA/1.0.0/index.html.haml` until your translation is ready.
- commit your changes to your own [fork][fork]
- submit a [Pull Request][pull-request] with your changes

It may take some time to review your submitted Pull Request. Try to involve a
few native speakers of the language you're translating to in the Pull Request
comments. They'll help review your translation for simple mistakes and give us
a better idea of whether your translation is accurate.

## Contribute

Please do contribute! Issues and pull requests are welcome.

Thank you for your help improving software one changelog at a time!

[changelog]: ./CHANGELOG.md
[changelog-badge]: https://img.shields.io/badge/changelog-Keep%20a%20Changelog%20v1.1.0-%23E05735
[license]: ./LICENSE
[rbenv]: https://github.com/rbenv/rbenv
[ruby-version]: .ruby-version
[source]: source/
[pull-request]: https://help.github.com/articles/creating-a-pull-request/
[fork]: https://help.github.com/articles/fork-a-repo/
[version-badge]: https://img.shields.io/badge/version-1.1.0-blue.svg
[license-badge]: https://img.shields.io/badge/license-MIT-blue.svg
```

## File: source/ar/1.0.0/index.html.haml
```haml
---
description: لا تدع أصدقائك يخلطون سجلات git مع سجلات التغيير
title: احتفظ بسجل التغيير
language: ar
version: 1.0.0
---
<link type="text/css" rel="stylesheet" href="https://cdn.rawgit.com/rastikerdar/vazir-font/v19.0.0/dist/font-face.css">

<style> 
body,html,h1,h2,h3,h4,h5,h6,a{font-family:Vazir;direction:rtl;text-align:right} 
div.frequently-asked-questions h4:after{float:left} 
pre {direction:ltr;text-align:left} 
p.yanked {direction:ltr;text-align:right}
</style> 

.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description
    
  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    ما هو سجل التغيير؟

  %p
    سجل التغيير هو ملف يحتوي على قائمة مرتبة ترتيبًا زمنيًا بالتغييرات البارزة لكل إصدار من المشروع.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    لماذا الاحتفاظ بسجل التغيير؟

  %p
    لتسهيل الأمر على المستخدمين والمساهمين أن يروا بدقة التغييرات البارزة التي تم إجراؤها بين كل إصدار (أو نسخة) من المشروع.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    من يحتاج إلى سجل التغيير؟

  %p
    الناس على العموم. سواء أكان المستهلكون أم المطورون ، فإن المستخدمين النهائيين للبرامج هم بشر يهتمون بما يوجود في البرنامج ، وعندما يتغير فإنهم يرغبون في معرفة السبب والكيف.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    كيف أصنع سجل تغيير جيد؟

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    مبادئ إرشادية

  %ul
    %li
      سجلات التغيير موجهة <em> للبشر </em> وليس للآلات.
    %li
      يجب أن يكون هناك فصل لكل نسخة.
    %li
      يجب تجميع كل التغييرات المتشابهة.
    %li
      يجب أن تكون النسخ والأقسام قابلة للربط.
    %li
      أحدث نسخة تأتي أولا.
    %li
      يتم عرض تاريخ إصدار كل نسخة.
    %li
      اذكر ما إذا كنت تتبع #{link_to "الأصدرة الدلالية" , data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types أنواع التغييرات

  %ul
    %li
      %code Added
      للحصول على ميزات جديدة.
    %li
      %code Changed
      للتغييرات في الوظائف الحالية.
    %li
      %code Deprecated
      للميزات التي ستتم إزالتها قريبًا.
    %li
      %code Removed
      للميزات التي تمت إزالتها الآن.
    %li
      %code Fixed
      لأي إصلاحات للأعطال البرمجية.
    %li
      %code Security
      في حالة وجود ثغرات أمنية.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    كيف يمكنني تقليل الجهد المطلوب لصيانة سجل التغيير؟

  %p
    احتفظ بقسم 
    <code> لم يتم طرحه (Unreleased)</code> 
    بالأعلى لتتبع التغييرات القادمة.

  %p يخدم هذا غرضين:

  %ul
    %li
      يمكن للأشخاص رؤية التغييرات التي يتطلعون لها في الإصدارات القادمة

    %li
      في وقت الإصدار ، يمكنك نقل تغييرات قسم <code> لم يتم طرحه (Unreleased)</code> إلى قسم إصدار النسخة الجديدة.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    هل يمكن أن تكون سجلات التغيير سيئة؟

  %p نعم. فيما يلي بعض الأمثلة التي يمكن أن تكون فيها سجلات التغيير عديمة فائدة.
  
  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    تضمين  تبديلات السجلات

  %p
    
    استخدام تضمينات 
    Git (commits)
    كسجلات التغيير هو فكرة سيئة: فهي مليئة بالضوضاء. أشياء مثل دمج التضمينات، تضمينات بعناوين غامضة، تغييرات التوثيق وما إلى ذلك
    

  %p
    الغرض من التضمين هو توثيق خطوة في تطوير شفرة المصدر. بعض المشاريع تنظف تضميناتها ، وبعضها الآخر لا يفعل ذلك.

  %p
    الغرض من فصول سجلات التغيير هو توثيق الاختلافات البارزة بين نسختين ، غالبًا عبر تضمينات عديدة ، لإيصالها بوضوح إلى المستخدمين النهائيين.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    تجاهل التخريدات (Deprecations)

  %p
    عندما تتم الترقية من إصدار إلى آخر ، يجب أن يكون جليا متى ينكسر شيء ما. يجب أن يمكن الترقية إلى إصدار يعد التخريدات ، وإزالة ما تم تخريده ، ثم الترقية إلى الإصدار حيث يصبح التخريد عملية إزالة وظائف.
  %p
    إذا لم تفعل شيئًا آخر ، فقم بعد التخريدات، عمليات الإزالة وأي تغييرات مُعطبة للبرنامج في سجل التغيير الخاص بك.
  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    تواريخ مربكة

  %p
    تتنوع تنسيقات التاريخ الإقليمية في جميع أنحاء العالم وغالبًا ما يكون من الصعب العثور على تنسيق تاريخ يُشعر أنه بديهي للجميع. تتمثل ميزة التواريخ المنسقة مثل 
    <code> 2017-07-17 </code>
    في أنها تتبع ترتيب الوحدات الأكبر إلى الأصغر: السنة والشهر واليوم. لا يتداخل هذا التنسيق بطرق غامضة مع تنسيقات التاريخ الأخرى ، على عكس بعض التنسيقات الإقليمية التي تغير موضع أرقام الشهر واليوم.
    بما أن تنسيق التاريخ هذا هو 
    #{link_to "standard ISO" , data.links.isodate}
    فإنه تنسيق التاريخ الموصى به لفصول سجلات التغيير.

  %aside
    هناك المزيد. ساعدني في جمع هذه الأنماط المضادة عن طريق #{link_to "فتح مشكل", data.links.issue}  أو طلب السحب.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    أسئلة شائعة
  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    هل هناك تنسيق قياسي لسجلات التغيير؟

  %p
    ليس حقا. 
    هناك إرشادات
    #{link_to "GNU changelog style guide", data.links.gnustyle} أو #{link_to "two-paragraph-long GNU NEWS file", data.links.gnunews} . 
    كلاهما إما غير كافٍ أو غير مناسب.
  %p
    يهدف هذا المشروع إلى أن يكون 
    = link_to "تحسين الصيغة الاصلاحية لسجلات التغيير.", data.links.changelog
    تمت صياغته من مراقبة الممارسات الجيدة في مجتمع المصادر المفتوحة وجمعها.
  %p
    كل من النقد السليم والمناقشة والاقتراحات للتحسين
    = link_to "مرحبٌ بها" , data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    كيف يجب تسمية ملف سجلات التغيير؟

  %p
    أطلق عليه اسم 
    <code> CHANGELOG.md </code>. 
    تستخدم بعض المشاريع التسميات التالية 
    <code> HISTORY </code> أو <code> NEWS </code> أو <code> RELEASES </code>.

  %p
    بما أن اسم ملف سجل التغيير الخاص بك لا يهم كثيرًا ، فلماذا تجعل من الصعب على المستخدمين النهائيين العثور على التغييرات البارزة؟

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    وماذا عن إصدارات 
    GitHub؟

  %p
    إنها مبادرة رائعة.
    يمكن استخدام #{link_to "الإصدارات", data.links.github_releases} 
    لتحويل علامات 
    git 
    البسيطة 
    (  على سبيل المثال علامة باسم <code> v1.0.0 </code> ) 
    إلى ملاحظات إصدار منسقة عن طريق إضافة ملاحظات الإصدار يدويًا أو يمكنها سحب رسائل علامة 
    git 
    المشروحة وتحويلها إلى ملاحظات.
  %p
    تُنشئ إصدارات 
    GitHub
    سجل تغيير غير محمول والذي لا يمكن عرضه إلا للمستخدمين في سياق 
    GitHub.
    من الممكن جعلها تشبه إلى حد كبير تنسيق 
    Keep a Changelog ، 
    لكنها تميل إلى أن تكون أكثر مشاركة قليلاً.
  %p
    يمكن القول أيضًا أن الإصدار الحالي من إصدارات 
    GitHub 
    يصعب استعماله من قبل المستخدمين النهائيين ، على عكس الملفات النموذجية ذات الأحرف الكبيرة 
    (وما إلى ذلك  <code> README </code> و <code> CONTRIBUTING </code>)
    هناك مشكلة ثانوية أخرى وهي أن الواجهة لا تقدم حاليًا روابط لتضمين السجلات بين كل إصدار.
  
  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    هل يمكن التحليل النحوي لسجلات التغيير تلقائيًا؟

  %p
    يصعب فعل ذلك لأن الأشخاص يتبعون تنسيقات وأسماء ملفات مختلفة تمامًا.

  %p
    #{link_to "Vandamme", data.links.vandamme} 
    هو جوهرة 
    Ruby 
    تم إنشاؤها بواسطة فريق 
    Gemnasium 
    والتي تحلل نحويا العديد من (ولكن ليس كل) سجلات تغيير المشروع مفتوحة المصدر.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    وماذا عن الإصدارات المسحوبة 
    "Yanked"؟

  %p
    الإصدارات المسحوبة هي نسخ حذفت بسبب عطب خطير أو مشكلة أمنية. غالبًا لا تظهر هذه الإصدارات في سجلات التغيير ولكن يجب فعل ذلك. هذه هي الطريقة التي يجب أن تعرضهم بها:

  %p.yanked  <code>## [0.0.5] - 2014-12-13 [YANKED]</code> 

  %p
    تم تسليط الضوء على علامة
    <code> [YANKED] </code> 
    لسبب وجيه.إذ من المهم أن يلاحظها الناس. نظرًا لأنها محاطة بأقواس ، فمن السهل أيضًا القيام بالتحليل النحوي لها برمجيًا.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    هل يجب عليك إعادة كتابة سجل التغيير؟

  %p
    بالتأكيد. هناك دائمًا أسباب وجيهة لتحسين سجلات التغيير. أقوم بفتح طلبات السحب بانتظام لإضافة الإصدارات المفقودة إلى مشاريع مفتوحة المصدر مع سجلات التغيير المتروكة.

  %p
    من الممكن أيضًا أن تكتشف أنك نسيت معالجة تغيير مُعطِب في ملاحظات النسخة. من الواضح أنه مهم بالنسبة لك تحديث سجل التغيير في هذه الحالة.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    كيف يمكنني المساهمة؟

  %p
    هذه الوثيقة ليست 
    <strong> الحقيقة </strong>
    ؛ إنه رأيي المدروس بعناية ، إلى جانب المعلومات والأمثلة التي جمعتها.

  %p
    هذا لأنني أريد أن يتوصل مجتمعنا إلى إجماع. أعتقد أن المناقشة لا تقل أهمية عن النتيجة النهائية.

  %p
    لذا يرجى <strong>#{link_to "المشاركة", data.links.repo}</strong>.

.press
  %h3 محادثات
  %p
    حضرت  #{link_to "بودكاست سجلات التغيير" , data.links.thechangelog} للحديث عن سبب اهتمام المشرفين والمساهمين بسجلات التغيير ، وكذلك عن الدوافع وراء هذا المشروع.
```

## File: source/assets/images/keep-a-changelog-mark.svg
```xml
<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 187.05 184.89"><defs><style>.cls-1{fill:#f4e0c2;}</style></defs><title>Artboard 1 copy 3</title><path class="cls-1" d="M62.09,6.5c-15.39,3.08-27.7,10.71-36.6,22.67C14.11,44.48,7.71,63.39,5.36,88.69c-.92,10,.06,20.05,1.34,31.18a73.74,73.74,0,0,0,15.86,38.56c6.9,8.56,14.82,14.18,24.2,17.19a88.55,88.55,0,0,0,24.67,4.25,45.42,45.42,0,0,0,5.88-.25l3.08-.31c4.24-.43,8.49-.86,12.73-1.37a158.36,158.36,0,0,0,54.68-16.7c14.84-7.75,25.22-19.18,30.84-34,2.62-6.88,3.87-12.67,3.87-18.08a33.91,33.91,0,0,0-.47-5.64c-3-17.59-9.14-32.75-18.9-46.33-12.09-16.83-24.32-28.68-38.48-37.31C103.11,6.76,82.64,2.38,62.09,6.5Zm29.73,8.13c20.19,4.42,38,14.81,52.92,30.87,16.92,18.2,26.56,37.31,29.49,58.42.53,3.91-.09,8.24-.71,12-2.87,17.21-12.3,30.05-28,38.16a155.31,155.31,0,0,1-53,16.14c-4.13.49-8.28.91-13,1.39l-.65.07a50.59,50.59,0,0,1-12.26-.23h0c-5.11-.72-10.89-1.55-16.45-2.87-13.07-3.11-22.7-12-29.42-27.06-5.06-11.37-7.68-24-8-38.7a132.53,132.53,0,0,1,8.5-50.28C25.83,40.4,32.46,27,46.85,19.86,61,12.82,75.74,11.11,91.83,14.63Z"/><path class="cls-1" d="M39.83,44.82C28.74,60.48,23,78.7,22.74,99a74.87,74.87,0,0,0,5.94,32.45c8,19,22.58,30.06,42.21,32,20.68,2,41-2.82,60.49-14.33a60.07,60.07,0,0,0,20.5-19.41,52.54,52.54,0,0,0,8.64-28.54c0-16-7.51-33.07-22.55-50.76a47.36,47.36,0,0,0-5.21-5.23c-23-19.78-44.72-25.81-66.87-18.4C54.05,30.67,45.78,36.43,39.83,44.82ZM148.43,121a50.36,50.36,0,0,1-21.5,22.1c-16.67,9.11-31.92,13.14-47.77,12.67H79c-11.47.45-21.38-2.36-30.31-8.6-5.18-3.61-9.18-8.89-12.24-16.15A80.91,80.91,0,0,1,30,99.42a93.56,93.56,0,0,1,7.57-35.9,89.91,89.91,0,0,1,6.47-12l1.2-2C50.2,41.33,58,37.4,68.19,34c16-5.27,31.72-2.55,49.55,8.56C130.93,50.82,141.12,63,148,78.87,154.56,93.81,154.69,108,148.43,121Z"/><path class="cls-1" d="M104.8,48c-17.06-7.52-32.45-5.88-45.72,4.88C48.76,61.24,42.58,73,39.63,89.81a53.56,53.56,0,0,0,5,34.12c7.85,15.19,20.7,23,37.17,22.56h.06c21.88-1.15,37.79-9.24,48.62-24.76a41.46,41.46,0,0,0,7.84-24,43,43,0,0,0-1.79-12.12C131.2,67.6,120.54,54.95,104.8,48ZM81.6,138.55h-.29c-7.43.59-14.75-1.67-21.77-6.73A27.13,27.13,0,0,1,50,119.1a56.51,56.51,0,0,1-3.8-20.17,62.68,62.68,0,0,1,5.52-25c5-11.51,12.75-18.51,23.78-21.4,9.07-2.36,18.1-1.45,27.59,2.8,13.78,6.16,22.73,17.25,26.62,33s-2,31.1-15.28,40.11C103.49,135.83,92.75,139.14,81.6,138.55Z"/><path class="cls-1" d="M83.47,105.25v14.12H74.93V68.46h8.54V96.28l9.11-9.89h11.16L89.9,100.87l16.81,18.5H95.9Z"/></svg>
```

## File: source/assets/images/logo.svg
```xml
<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 545.64 184.89"><defs><style>.cls-1{fill:#f15d30;}</style></defs><title>Artboard 1</title><path class="cls-1" d="M217.72,63.21V77.79H208.9V25.23h8.82V54l9.4-10.21h11.52L224.35,58.69l17.35,19.1H230.55Z"/><path class="cls-1" d="M275.38,63.72h-26c.58,4.52,4.37,7.29,9.4,7.29,3.43,0,6.93-1.46,8.53-4.37,2.11,1.17,4.59,2.48,6.78,3.64-3.06,5.83-9.62,8.24-15.89,8.24-9.7,0-17.64-7.14-17.64-17.86S248.55,43,258.25,43s17.28,6.93,17.28,17.64C275.53,61.54,275.45,62.92,275.38,63.72Zm-8.53-6.05c-.66-4.67-4.16-7.22-8.53-7.22-4.67,0-8.31,2.62-9,7.22Z"/><path class="cls-1" d="M313.87,63.72h-26c.58,4.52,4.37,7.29,9.4,7.29,3.43,0,6.93-1.46,8.53-4.37,2.11,1.17,4.59,2.48,6.78,3.64-3.06,5.83-9.62,8.24-15.89,8.24-9.7,0-17.64-7.14-17.64-17.86S287,43,296.74,43,314,49.94,314,60.66C314,61.54,313.94,62.92,313.87,63.72Zm-8.53-6.05c-.66-4.67-4.16-7.22-8.53-7.22-4.67,0-8.31,2.62-9,7.22Z"/><path class="cls-1" d="M328.59,48.34C330.41,45,334.79,43,339.23,43c8.89,0,16.62,7,16.62,17.71s-7.73,17.79-16.62,17.79c-4.45,0-8.82-1.82-10.64-5.25V94.34H319.7V43.75h8.89Zm9.19,21.72a8.83,8.83,0,0,0,9-9.18,9,9,0,1,0-18,0A8.8,8.8,0,0,0,337.77,70.07Z"/><path class="cls-1" d="M404,73.42c-2.19,3.5-7.22,5-10.57,5a17.24,17.24,0,0,1-17.35-17.71A17.19,17.19,0,0,1,393.46,43c3.06,0,8.09,1.17,10.57,5V43.75h8.89v34H404Zm-9.33-3.5c4.81,0,9.33-3.35,9.33-9.19s-4.67-9.18-9.33-9.18a9.19,9.19,0,1,0,0,18.37Z"/><path class="cls-1" d="M225.08,113.83a17.57,17.57,0,0,1,14.14,6.63l-7.07,5.32a8.71,8.71,0,0,0-7.07-3.35c-5,0-9.55,3.5-9.55,9a9.39,9.39,0,0,0,9.55,9.33,8.77,8.77,0,0,0,7.07-3.43l7.14,5.25a17.68,17.68,0,0,1-14.22,6.71c-9.77,0-18.37-7.14-18.37-17.86S215.31,113.83,225.08,113.83Z"/><path class="cls-1" d="M253.87,148.6H245V96h8.89v23.77c1.6-4.16,7.22-6,10.57-6,8.82,0,13.71,5.83,13.71,15.82v19h-8.89V130.09A7.18,7.18,0,0,0,262,122.5c-3.72,0-8.16,2-8.16,7.95Z"/><path class="cls-1" d="M311.75,144.23c-2.19,3.5-7.22,5-10.57,5a17.72,17.72,0,0,1,0-35.43c3.06,0,8.09,1.17,10.57,5v-4.23h8.89v34h-8.89Zm-9.33-3.5c4.81,0,9.33-3.35,9.33-9.19s-4.67-9.18-9.33-9.18a9.19,9.19,0,1,0,0,18.37Z"/><path class="cls-1" d="M337.77,148.6h-8.89v-34h8.89v5c1.9-3.64,6.2-5.76,10.13-5.76,7.95,0,12.68,5,12.68,14.65V148.6H351.7V130.09c0-5-2.62-7.58-6.49-7.58s-7.44,2-7.44,7.95Z"/><path class="cls-1" d="M383.84,157.5c8.6,0,10.86-6.42,10.35-13.56-1.75,3.43-6.42,5.39-10.86,5.39-8.89,0-17.06-6.93-17.06-17.64,0-10.94,8.17-17.86,17.06-17.86,4.45,0,9,1.9,10.86,5.32v-4.59h8.68v29.16c0,15.16-7.22,21.87-19,21.87a17.81,17.81,0,0,1-16.26-10.06c2-.95,5.25-2.7,7.22-3.64C376.26,155.24,380.27,157.5,383.84,157.5Zm.87-35.14a9.26,9.26,0,0,0,0,18.52,9.23,9.23,0,0,0,9.26-9.33C394,125.64,389.6,122.36,384.72,122.36Z"/><path class="cls-1" d="M443.76,134.53h-26c.58,4.52,4.37,7.29,9.4,7.29,3.43,0,6.93-1.46,8.53-4.37,2.11,1.17,4.59,2.48,6.78,3.64-3.06,5.83-9.62,8.24-15.89,8.24-9.7,0-17.64-7.14-17.64-17.86s7.95-17.64,17.64-17.64,17.28,6.93,17.28,17.64C443.91,132.35,443.83,133.73,443.76,134.53Zm-8.53-6.05c-.66-4.67-4.16-7.22-8.53-7.22-4.67,0-8.31,2.62-9,7.22Z"/><path class="cls-1" d="M449.59,96h8.89V148.6h-8.89Z"/><path class="cls-1" d="M464.6,131.47c0-10.72,8-17.64,17.71-17.64S500,120.75,500,131.47s-8.09,17.79-17.71,17.79S464.6,142.19,464.6,131.47Zm26.61,0a8.89,8.89,0,0,0-17.79,0,8.9,8.9,0,1,0,17.79,0Z"/><path class="cls-1" d="M521.61,157.5c8.6,0,10.86-6.42,10.35-13.56-1.75,3.43-6.42,5.39-10.86,5.39-8.89,0-17.06-6.93-17.06-17.64,0-10.94,8.17-17.86,17.06-17.86,4.45,0,9,1.9,10.86,5.32v-4.59h8.68v29.16c0,15.16-7.22,21.87-19,21.87a17.81,17.81,0,0,1-16.26-10.06c2-.95,5.25-2.7,7.22-3.64C514,155.24,518,157.5,521.61,157.5Zm.87-35.14a9.26,9.26,0,0,0,0,18.52,9.23,9.23,0,0,0,9.26-9.33C531.74,125.64,527.37,122.36,522.48,122.36Z"/><path class="cls-1" d="M62.09,6.5c-15.39,3.08-27.7,10.71-36.6,22.67C14.11,44.48,7.71,63.39,5.36,88.69c-.92,10,.06,20.05,1.34,31.18a73.74,73.74,0,0,0,15.86,38.56c6.9,8.56,14.82,14.18,24.2,17.19a88.55,88.55,0,0,0,24.67,4.25,45.42,45.42,0,0,0,5.88-.25l3.08-.31c4.24-.43,8.49-.86,12.73-1.37a158.36,158.36,0,0,0,54.68-16.7c14.84-7.75,25.22-19.18,30.84-34,2.62-6.88,3.87-12.67,3.87-18.08a33.91,33.91,0,0,0-.47-5.64c-3-17.59-9.14-32.75-18.9-46.33-12.09-16.83-24.32-28.68-38.48-37.31C103.11,6.76,82.64,2.38,62.09,6.5Zm29.73,8.13c20.19,4.42,38,14.81,52.92,30.87,16.92,18.2,26.56,37.31,29.49,58.42.53,3.91-.09,8.24-.71,12-2.87,17.21-12.3,30.05-28,38.16a155.31,155.31,0,0,1-53,16.14c-4.13.49-8.28.91-13,1.39l-.65.07a50.59,50.59,0,0,1-12.26-.23h0c-5.11-.72-10.89-1.55-16.45-2.87-13.07-3.11-22.7-12-29.42-27.06-5.06-11.37-7.68-24-8-38.7a132.53,132.53,0,0,1,8.5-50.28C25.83,40.4,32.46,27,46.85,19.86,61,12.82,75.74,11.11,91.83,14.63Z"/><path class="cls-1" d="M39.83,44.82C28.74,60.48,23,78.7,22.74,99a74.87,74.87,0,0,0,5.94,32.45c8,19,22.58,30.06,42.21,32,20.68,2,41-2.82,60.49-14.33a60.07,60.07,0,0,0,20.5-19.41,52.54,52.54,0,0,0,8.64-28.54c0-16-7.51-33.07-22.55-50.76a47.36,47.36,0,0,0-5.21-5.23c-23-19.78-44.72-25.81-66.87-18.4C54.05,30.67,45.78,36.43,39.83,44.82ZM148.43,121a50.36,50.36,0,0,1-21.5,22.1c-16.67,9.11-31.92,13.14-47.77,12.67H79c-11.47.45-21.38-2.36-30.31-8.6-5.18-3.61-9.18-8.89-12.24-16.15A80.91,80.91,0,0,1,30,99.42a93.56,93.56,0,0,1,7.57-35.9,89.91,89.91,0,0,1,6.47-12l1.2-2C50.2,41.33,58,37.4,68.19,34c16-5.27,31.72-2.55,49.55,8.56C130.93,50.82,141.12,63,148,78.87,154.56,93.81,154.69,108,148.43,121Z"/><path class="cls-1" d="M104.8,48c-17.06-7.52-32.45-5.88-45.72,4.88C48.76,61.24,42.58,73,39.63,89.81a53.56,53.56,0,0,0,5,34.12c7.85,15.19,20.7,23,37.17,22.56h.06c21.88-1.15,37.79-9.24,48.62-24.76a41.46,41.46,0,0,0,7.84-24,43,43,0,0,0-1.79-12.12C131.2,67.6,120.54,54.95,104.8,48ZM81.6,138.55h-.29c-7.43.59-14.75-1.67-21.77-6.73A27.13,27.13,0,0,1,50,119.1a56.51,56.51,0,0,1-3.8-20.17,62.68,62.68,0,0,1,5.52-25c5-11.51,12.75-18.51,23.78-21.4,9.07-2.36,18.1-1.45,27.59,2.8,13.78,6.16,22.73,17.25,26.62,33s-2,31.1-15.28,40.11C103.49,135.83,92.75,139.14,81.6,138.55Z"/><path class="cls-1" d="M83.47,105.25v14.12H74.93V68.46h8.54V96.28l9.11-9.89h11.16L89.9,100.87l16.81,18.5H95.9Z"/></svg>
```

## File: source/assets/javascripts/all.js
```javascript
document.addEventListener("DOMContentLoaded", function(){
  var select = document.querySelector('.locales select');
  select.addEventListener('change', function(event){
    origin = window.location.origin;
    languageCode = event.currentTarget.selectedOptions[0].value
    window.location.replace(origin + "/" + languageCode)
  });
});
```

## File: source/assets/stylesheets/_mixins.sass
```sass
// Via: https://medium.com/codeartisan/breakpoints-and-media-queries-in-scss-46e8f551e2f2

// Small devices
@mixin sm
  @media (min-width: #{$screen-sm-min})
    @content

// Medium devices
@mixin md
  @media (min-width: #{$screen-md-min})
    @content

// Large devices
@mixin lg
  @media (min-width: #{$screen-lg-min})
    @content

// Extra large devices
@mixin xl
  @media (min-width: #{$screen-xl-min})
    @content
```

## File: source/assets/stylesheets/_variables.sass
```sass
$base-font-family: 'Muli', "Helvetica Neue", Helvetica, Arial, sans-serif
$source-code-font-family: "Source Code Pro", monospace

$color-black: #342828
$color-white: #FFFFFF
$color-ocre: #E05735
$color-gold: #faa930
$color-bark: #3F2B2D
$color-sand: #FEECD3
$color-light-sand: lighten($color-sand, 10%)

/* === Breakpoints === */

// Small tablets and large smartphones (landscape view)
$screen-sm-min: 576px

// Small tablets (portrait view)
$screen-md-min: 768px

// Tablets and small desktops
$screen-lg-min: 992px

// Large tablets and desktops
$screen-xl-min: 1200px
```

## File: source/assets/stylesheets/anchors.sass
```sass
@import "variables"

.anchor
  display: none
  font-style: normal
  font-weight: normal
  position: absolute
  right: 100%
  top: 0

.anchor::before
  line-height: 3.2
  content: "∞"
  color: $color-ocre
  padding-right: 0.1em

.anchor:hover
  a
    text-decoration: none
```

## File: source/assets/stylesheets/application.css.sass
```sass
@import "variables"
@import "mixins"
@import "layout"
@import "anchors"
@import "sections"

html
  font: 16px/1.4 $base-font-family
  color: #342828
  background-color: $color-sand

a
  color: $color-ocre
  text-decoration: none

a:hover
  text-decoration: underline

h1
  color: $color-ocre
  font-size: 3.2em
  font-weight: bold
  font-family: $base-font-family
  margin-bottom: 0
  line-height: 1
  text-transform: lowercase

h2
  font-size: 1.2em
  margin-top: 0.2em
  margin-bottom: 2em

h1, h2
  text-align: left

h3
  font-size: 1.3em
  font-family: $base-font-family
  margin-bottom: 1em
  position: relative
  padding-left: .1em

h3:hover .anchor, h3:focus .anchor
  display: block

h3 ~ p
  margin-top: 0

h1, h2, h3
  margin-top: 0
  padding-top: 1em

.about
  text-align: center

.about
  margin-top: 0

header
  .mark
    float: left
    margin-top: 2em
    margin-left: 2em
    clear: both
    width: 80px

    @inclde sm
      margin-top: 1em
      margin-left: 1.5em
      width: 100px

    @include md
      width: 120px
      margin-top: 3em 
      margin-bottom: 2em 
      margin-left: 3em

    @include lg
      width: 150px
      clear: none
      margin-top: 3em 
      margin-bottom: 4em 

    @include xl
      width: 150px

  .locales
    margin-top: 2em
    margin-right: 3em
    margin-left: 1em
    font-size: 75%

    @include md
      font-size: 85%
  
    @include lg
      font-size: 100%
      margin-bottom: -2em
      margin-top: 2em

  .newer, .last-version-notice
    padding: 1rem
    background-color: rgba(0, 0, 0, 1)
    color: #efefef

.footer
  line-height: 2.2

  .mark
    float: left

.locales
  float: right
  background-color: rgba(255, 255, 255, .5)
  border-radius: .5rem
  padding: .7rem 1rem

.locales li
  float: right
  font-size: 0.7em
  display: inline
  list-style-type: none
  white-space: nowrap

pre, code
  background-color: $color-light-sand
  font-family: $source-code-font-family
code
  white-space: nowrap
```

## File: source/assets/stylesheets/layout.sass
```sass
@import "variables"
@import "mixins"

$break-small: em(480px)
$break-medium: em(800px)
$break-large: em(1024px)

body
  margin: 0

article
  margin: 0 auto
  max-width: 65em

.main
  background-color: $color-white

  div
    padding-left: 8%
    padding-right: 8%
    padding-bottom: 3em

  div.title

    aside
      background-color: lighten(desaturate($color-bark, 5%), 10%)
      margin-bottom: -3em
      margin-left: -5%
      margin-right: -5%
      margin-top: 3em
      padding: 2em 5%
      text-align: center

    aside &
      background-position: top

article p
  font-size: 1.3em

article code
  border-radius: 0.315em
  border: 1px solid #ccc
  padding: 0 0.3em 0.040em
  font-size: 0.9em

article ul
  font-size: 1.1em
  line-height: 1.5
  list-style-type: square
  padding-left: 1em

@include xl
  body
    margin: 8px

  .main
    margin-bottom: 2em

    div:not(.header)
      padding-left: 20%
      padding-right: 20%

      aside
        margin-left: -35%
        margin-right: -35%
        padding: 2em 20%
```

## File: source/assets/stylesheets/legacy.css.sass
```sass
$base-font-family: "Carrois Gothic", "Helvetica Neue", Helvetica, Arial, sans-serif
$source-code-font-family: "Source Code Pro", monospace

html
  font: 14px/1.4 $base-font-family
  color: #333

a
  color: #E10FCE
  font-weight: bold
  text-decoration: none

a:hover
  text-decoration: underline

h1
  color: #C647BF
  font-size: 4.1em
  font-weight: bold
  font-family: $source-code-font-family
  margin-bottom: 0
  line-height: 1
  word-spacing: -0.3em

h2
  margin-top: 0
  margin-bottom: 2em

h1, h2
  text-align: center

h3
  font-size: 1.3em
  font-family: $source-code-font-family
  margin-bottom: 0
  position: relative
  padding-left: .1em

h3:hover .anchor, h3:focus .anchor
  display: block

.anchor
  display: none
  font-style: normal
  font-weight: normal
  position: absolute
  right: 100%
  top: 0

.anchor::before
  content: '¶'

.anchor:hover
  text-decoration: none

h3 ~ p
  margin-top: 0

article
  margin: 0 auto
  width: 640px

article p
  font-size: 1.3em

article code
  border-radius: 0.315em
  border: 1px solid #ccc
  padding: 0 0.3em 0.040em
  font-size: 0.9em

article img
  margin: 0 auto
  max-width: 100%
  text-align: center
  display: block
  box-shadow: rgba(0, 0, 0, 0.37) 0px 0px 6px
  border-radius: 7px
  padding: 0.4em 0.9em

article > ul
  font-size: 1.2em
  padding-left: 0

article > ul ul
  padding-left: 1em

article ul
  line-height: 1.5
  list-style-position: inside
  list-style-type: square

footer
  font-size: .7em
  border-top: 1px solid #e9e6e1
  margin-top: 1em
  margin-bottom: 2em
  padding-top: 1em

.license
  float: left

.about
  float: right

.about, .license
  margin-top: 0

.changelog
  border: 1px solid #aaa
  margin: 0 0.5em
  padding: 1em
  overflow-x: auto
  height: 20em

.locales li
  display: inline
  list-style-type: none
  padding-right: 20px
  white-space: nowrap

pre, code
  font-family: $source-code-font-family

.newer, .last-version-notice
  text-align: center
  margin: 1em
```

## File: source/assets/stylesheets/sections.sass
```sass
@import "variables"
@import "mixins"

div.header
  padding-top: 5em
  padding-bottom: 0.1em
  background-color: $color-ocre
  background-image: image-url("bg-hero@2x.png")
  background-size: cover

  h1
    color: $color-sand

  h2
    font-weight: normal

  a
    color: $color-black

  .title
    width: 60%
    margin: 0 auto
    margin-bottom: -2em
    font-size: 60%
    margin-left: 10em
    margin-right: 2em

    @include sm
      width: 60%
    @include md
      font-size: 100%
      margin-left: 7em
      margin-right: 2em
    @include lg
      margin-left: auto

  .changelog
    background-color: $color-white
    box-shadow: 0px 4px 12px 0px hsla(0, 0%, 0%, 0.2)
    font-size: 0.8em
    height: 20em
    margin-bottom: -14em
    margin-left: 0.5em
    margin-right: 0.5em
    margin-top: 1em
    overflow-x: auto
    padding: 1em

div.answers
  margin-top: 12em
  background-color: $color-white

  h3
    font-size: 1.7em
    color: $color-ocre

  a
    color: $color-gold

  p
    font-size: 1em

.good-practices
  background-color: $color-gold
  background-image: image-url("bg-img1@2x.png")
  background-position: bottom left
  background-repeat: no-repeat
  background-size: 100%
  padding-top: 2em

  a.anchor
    color: $color-black

  h3
    font-size: 1.6em
    font-weight: bold

  ul
    font-size: 1em

  p
    a
      color: $color-white
      text-decoration: underline

div.bad-practices
  color: $color-white
  background-color: $color-bark
  background-image: image-url("bg-img2@2x.png")
  background-position: bottom left
  background-repeat: no-repeat
  background-size: 100%
  padding-top: 3em

  code
    color: $color-black
    background-color: desaturate(lighten($color-bark, 50%), 10%)
    border: none

  p, aside
    a
      color: $color-gold
      text-decoration: underline

  aside
    a
      text-decoration: none

  h3
    font-size: 1.7em
    color: $color-gold
    float: left


  p
    clear: left
    font-size: 1em

div.effort
  padding-top: 2em
  background-color: $color-white

  h3
    font-size: 1.6em
    font-weight: bold
    color: $color-ocre

div.frequently-asked-questions
  padding-top: 2em
  background-color: $color-white

  h2
    color: $color-ocre

  h3
    color: $color-ocre
    margin-bottom: 1.5em
    font-size: 1.7em

  h4
    cursor: pointer
    border-bottom: 1px solid transparentize($color-bark, 0.8)
    padding: 0 1.1em 1.3em 0

  h4 ~ p
    font-size: 1em
    margin-left: 1em

.footer
  font-size: .6em
  font-weight: normal
  border-top: 1px solid #e9e6e1
  padding-top: 1em
  padding-left: 2em
  padding-right: 2em

  background-color: $color-ocre
  color: $color-sand

  a
    color: $color-white
    text-decoration: underline

  &:after
    content: ""
    display: table
    clear: both

@include xl
  div.answers
    padding-left: 25%
    padding-right: 25%

  div.effort
    padding-left: 25%
    padding-right: 25%

  div.frequently-asked-questions
    padding-left: 25%
    padding-right: 25%

  .good-practices
    h3
      width: 50%
    h4
      padding-left: 15em
    ul
      padding-left: 16em

  .bad-practices
    h3
      width: 35%

    h4, p
      padding-left: 12em
```

## File: source/assets/stylesheets/table-of-contents.sass
```sass
@import "variables"

.toc
  position: fixed
  z-index: 9999
  top: 10%
  left: 0

  ul
    list-style-position: inside
    list-style-type: decimal-leading-zero
    padding: 0 0 0 1em
    li
      padding-left: 1em
      line-height: 2
      a
        padding: 0 1em
        line-height: 2
        display: none

        span
          display: none

    li:hover
      color: $color-sand
      background: $color-black
    li:hover a
      color: $color-sand
      font-weight: bold
      background-color: $color-black
      display: inline-block
```

## File: source/CNAME
```
keepachangelog.com
```

## File: source/cs/0.3.0/index.html.haml
```haml
---
title: Udržuj Changelog
description: Nenech své kamarády házet do CHANGELOGů™ git logy.
language: cs
version: 0.3.0
---

%h1= current_page.data.title

%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Co je to changelog?
  Changelog je soubor, který obsahuje organizovaný, chronologicky seřazený
  seznam podstatných změn pro každou verzi daného projektu.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Co je smyslem changelogu?
  Usnadnit uživatelům a přispěvatelům přesné zobrazení podstatných změn, které
  byly provedeny mezi jednotlivými vydáními (verzemi) daného projektu.

  ### Proč by mi na tom mělo záležet?
  Protože softwarové nástroje jsou pro lidi. Pokud ti na tom nezáleží, tak proč
  přispíváš do open source projektů? Určitě musí být v tvém úžasném malém mozku
  alespoň jádro (haha!) ochoty.

  [Bavil jsem se s Adamem Stacoviakem a Jerodem Santem na podcastu The Changelog][thechangelog]
  (název sedí, co?) o tom proč by na tom mělo vedoucím a přispěvatelům projektů
  záležet a o tom jaká motivace je za tímto projektem. Jestli máš chvilku času
  (1:06), stojí to za poslech.

  ### Co dělá changelog dobrým?
  Výborná otázka, díky za ni.

  Dobrý changelog se drří těchto principů:

  - Je psaný pro lidi, ne pro stroje, takže čitelnost je zásadní.
  - Dá se v něm jednoduše odkázat na libovolnou sekci (proto raději Markdown než plain text).
  - Jedna pod-sekce za verzi.
  - Seznam vydání ve zpětně-chronologickém pořadí (nejnovější navrchu).
  - Psaní všech dat ve formátu `RRRR-MM-DD`. (Příklad: `2012-06-02` pro `2. červen 2012`.) Je to mezinárodní, [rozumné](https://xkcd.com/1179/) a nezávislé na jazyce.
  - Explicitní uvedení toho, zda projekt dodržuje [Semantické Verzování][semver].
  - Každá verze by měla:
    - Uvádět datum vydání ve výše uvedeném formátu.
    - Seskupovat změny tak, aby popisovaly dopad na projekt, a to následovně:
      - `Added` pro nové funkce.
      - `Changed` pro změny v aktuálním fungování.
      - `Deprecated` pro dříve stabilní funkce, které budou odstraněny v novějších vydáních a nejsou podporovány.
      - `Removed` pro funkce odstraněné v daném vydání.
      - `Fixed` pro opravené chyby.
      - `Security` pro navržení aktualizace uživatelům v případě bezpečnostních problémů.


  ### Jak mohu vyžadovanou snahu snížit na minimum?
  Vždycky měj nahoře sekci `"Unreleased"`, ve které budou schromažďovány všechny
  změny.

  To plní hned dva účely:

  - Lidé uvidí, jaké změny mohou očekávat v následujících vydáních
  - V momentu vydání stačí změnit `"Unreleased"` na číslo verze a přidat nový nadpis `"Unreleased"` na vrch dokumentu.

  ### Co zaručeně rozbrečí jednorožce?
  Dobrá… Tak si to povíme.

  - **Zkopírování diffu commit logů.** To prostě nedělej, nikomu to nepomůže.
  - **Nezdůraznění dále nepodporovaných funcí.** Když lidé aktualizují na novou verzi, mělo by jim být bolestivě jasné, že se něco rozbije.
  - **Data v regionálních formátech.** V USA píšou lidé nejdříve měsíc ("06-02-2012" pro 2. června 2012, což nedává *vůbec žádný* smysl), zatímco mnoho lidí ve zbytku světa píše roboticky vypadající verzi "2 June 2012", ale vyslovují to jinak. "2012-06-02" je logicky napsané od největšího po nejmenší celek, nepřekrývá se nesrozumitelně s jinými fomáty data a je to [ISO standard](http://www.iso.org/iso/home/standards/iso8601.htm). Proto je to doporučovaný formát dat pro changelogy.

  To ale není všechno. Pomoz mi sbírat jednorožčí slzy tím že
  [otevřeš issue][issues] nebo pull request.

  ### Existuje pro formát changelogů nějaký standard?
  Bohužel ne. Klid. Vím, že se zuřivě snažíš najít ten odkaz na GNU návod jakým
  stylem psát changelog, nebo onen dvouodstavcový GNU NEWS soubor, který si říká
  "směrnice". Zmíněný GNU návod je sice hezký začátek, ale je smutně naivní. Být
  naivní není nic špatného, ale když lidé potřebují radu, málokdy to někomu
  pomůže. Obzvlášť, když existuje tolik situací a okrajových případů, se kterými
  se musí člověk nějak poprat.

  Tento projekt
  [obsahuje něco, co se doufám jednou stane lepší konvencí pro CHANGELOGy][CHANGELOG].
  Nemyslím si, že je momentální stav dostatečně dobrý, a jsem toho názoru, že
  jsme jako komunita schopni vymyslet lepší konvence, pokud se budeme snažit
  vybrat ty nejlepší praktiky z doopravdových softwarových projektů.
  Porozhlédněte se a mějte na paměti, že
  [diskuse a návrhy na zlepšení jsou vítány][issues]!

  ### Jak by se měl changelog jmenovat?
  Inu, pokud jste to už nepoznali z příkladu výše, `CHANGELOG.md` je zatím ta
  nejlepší konvence.

  Některé projekty používají `HISTORY.txt`, `HISTORY.md`, `History.md`,
  `NEWS.txt`, `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`,
  `releases.md`, apod.

  Je v tom binec. Všecha tato jména jen komplikují život lidem, kteří se ten
  dokument snaží najít.

  ### Proč lidé jednoduše nepoužívají `git log` diff?
  Protože diffy logů jsou plné šumu — přirozeně. Nebyly by vyhovujícím
  changelogem ani v hypotetickém projektu vedeném dokonalými lidmi, kteří nikdy
  nedělají překlepy, nikdy nezapomínají commitnout nové soubory a nikdy jim
  neunikne ani ta nejmenší část refactoringu. Cílem commitu je dokumentovat
  jeden miniaturní krok při procesu, ve kterém se kód vyvíjí z jedné podoby do
  druhé. Cílem changelogu je dokumentovat podstatné změny mezi těmito podobami.

  Stějně jako je rozdíl mezi dobrými komentáři a samotným kódem, je také rozdíl
  mezi changelogem a commitlogem: jeden říká *proč* a druhý jak.

  ### Mohou být changelogy automaticky parsované?
  Je to složité, jelikož se lidé uchylují k nejrůznějším formátům a názvům
  souborů.

  [Vandamme][vandamme] je Ruby gem vytvořený týmem [Gemnasium][gemnasium], který
  parsuje mnoho (ale ne všechny) changelogů open source projektů.

  ### Proč používáš jednou "CHANGELOG" a jindy "changelog"?
  "CHANGELOG" je název samotného souboru. Je to třichu křiklavé, ale to už je
  historická konvence, kterou se řídí mnoho open source projektů. Příklady
  podobných souborů mohou být [`README`][README], [`LICENSE`][LICENSE] a
  [`CONTRIBUTING`][CONTRIBUTING].

  Název psaný kapitálkami (díky kterému se ve starých operačních systémech
  soubory držely na nejvyšších pozicích) je používán za účelem upoutání
  pozornosti. Vzhledem k tomu, že se jedná o důležitá metadata o projektu a
  mohla by být užitečná pro kohokoliv, kdo ho chce používat nebo do něho
  přispívat, podobně [open source projektovým odznakům][shields].

  Když mluvím o "changelogu", myslím tím funkci tohoto souboru: logování změn.

  ### Jak potom vypadají ta vydání, která byla zpětně stažena?
  Zpětně stažená vydání jsou verze, které musely být zpětně odebrány kvůli vážné
  chybě nebo bezpečnostním rizikům. Tyto verze se často v changelogu ani
  neobjevují, ale měly by. Zobrazovat by se měli takto:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Tag `[YANKED]` je naschvál křiklavý. Je důležité, aby si ho lidé všimli. Díky
  tomu, že je v hranatých závorkách je take jednoduší ho parsovat programem.

  ### Měl by se někdy changelog přepisovat?
  Jasně. Vždy se najde dobrý důvod pro zlepšení changelogu. Sám často otevírám
  pull requesty pro přidání chybějících vydání v open source projektech s
  neudržovanými changelogy.

  Je také možné, že zjistíš, že v poznámkách k verzi ve tvém projektu není
  zmíněna změna, která něco mohla rozbít. V tom případě jě samozřejmě důležité,
  aby byl changelog aktualizován.

  ### Jak mohu přidat ruku k dílu?
  Tento dokument není čistá **pravda**; je to můj názor, nad kterým jsem opatrně
  uvažoval, v kombinaci s informacemi a příklady, které se mi podařilo získat.
  I přesto, že uvádím vlastní [CHANGELOG][] v [repozitáři na GitHubu][gh],
  naschvál jsem nevytvořil náležité *vydání* nebo jasný seznam pravidel, kterými
  se někdo má řídit (jako to například udělali na [SemVer.org][semver]).

  Je tomu tak proto, že chci aby komunita došla ke společné shodě. Já sám jsem
  toho názoru, že diskuse je důležitou součástí finálního výsledku.

  Takže prosím [**přispějte**][gh] svou trochou do mlýna.

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/cs/1.0.0/index.html.haml
```haml
---
title: Udržuj Changelog
description: Nenech své kamarády sypat git logy do changelogů.
language: cs
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Verze
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Co je to changelog?

  %p
    Changelog je soubor, který obsahuje organizovaný, chronologicky seřazený
    seznam podstatných změn pro každou verzi daného projektu.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Proč udržovat changelog?

  %p
    Aby uživatelé a přispěvatelé přesně věděli, jaké podstatné změny byly
    provedeny mezi jednotlivými vydáními (verzemi) daného projektu.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kdo potřebuje changelog?

  %p
    Lidé. Ať už se jedná o spotřebitele nebo vývojáře, koncoví uživatelé
    softwaru jsou lidská stvoření, kterým záleží na tom, co software obsahuje.
    Když se daný software změní, lidé chtějí vědět proč a jak.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Jak vytvořit dobrý changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Hlavní zásady

  %ul
    %li
      Changelogy jsou <em>pro lidi</em>, ne pro stroje.
    %li
      Changelog by měl obsahovat záznam pro každou verzi.
    %li
      Stejné typy změn by měly být seskupené.
    %li
      Měla by existovat možnost odkazovat na jednotlivé verze a sekce.
    %li
      Poslední verze je na prvním místě.
    %li
      U každé verze je poznamenáno datum jejího vydání.
    %li
      Zmiňte, zda se držíte #{link_to "Sémantického verzování", data.links.semver}

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Typy změn

  %ul
    %li
      %code Added
      pro nové funkce.
    %li
      %code Changed
      pro změny v existující funkcionalitě.
    %li
      %code Deprecated
      pro funkce, které budou brzy odstraněny.
    %li
      %code Removed
      pro odstraněné funkce.
    %li
      %code Fixed
      pro opravy chyb.
    %li
      %code Security
      v případě bezpečnostních zranitelností.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Jak minimalizovat úsilí potřebné k udržování changelogu?

  %p
    Udržováním <code>Unreleased</code> sekce na začátku souboru pro zaznamenávání
    nadcházejících změn.

  %p To plní hned dva účely:

  %ul
    %li
      Lidé uvidí, jaké změny mohou očekávat v následujících vydáních
    %li
      V čas vydání stačí přesunout změny z <code>Unreleased</code> sekce
      do sekce nového vydání.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Mohou být changelogy špatné?

  %p Ano. Zde je několik případů, kdy mohou být opakem užitečného.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffy z commit logu

  %p
    Používání diffů z commit logu jako changelogu je špatný nápad:
    jsou plné šumu. Obsahují věci jako merge commity, commity s
    nejasnými nadpisy, změny v dokumentaci, atd.

  %p
    Účelem commitu je zdokumentovat krok v evoluci zdrojového kódu.
    Některé projekty commity pročišťují, jiné zas ne.

  %p
    Účelem záznamu v changelogu je zdokumentovat podstatné změny,
    často napříč několika commity, a jasně je sdělit koncovým uživatelům.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorování odstraněných funkcí
  %p
    Když lidé upgradují z jedné verze na druhou, mělo by jim být bolestně
    jasné, když se něco rozbije. Mělo by být možné nejprve upgradovat na verzi,
    která oznámí, jaké funkce budou odstraněny, dané funkce odstranit
    a poté upgradovat na verzi, kde jsou zmíněné funkce již odstraněny.

  %p
    Když už nic jiného, je dobré alespoň vypsat odstraněné funkce a
    změny, které něco rozbíjí, do changelogu.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Matoucí data

  %p
    Regionální formáty dat se liší napříč světem a je často složité
    najít formát, který je přátelský a intuitivní pro všechny.
    Výhodou dat formátovaných jako <code>2017-07-17</code> je pořadí
    jednotek od největší po nejmenší: rok, měsíc a den. Tento formát
    se navíc nepřekrývá s jinými, narozdíl od některých regionálních
    formátů, které prohazují pozici měsíce a dne. Díky těmto důvodům,
    a také faktu, že je tento formát #{link_to "ISO standard", data.links.isodate},
    je doporučeným formátem pro záznamy v changelogu.

  %aside
    Je toho však víc. Pomozte mi sesbírat tyto antipatterny
    = link_to "otevřením issue", data.links.issue
    nebo pull requestu.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Časko kladené otázky

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Existuje pro formát changelogu nějaký standard?

  %p
    Ne. Je tu GNU stylová příručka pro changelog, nebo ta dvouodstavcová
    GNU "směrnice" pro NEWS soubor. Ani jedno však není vhodné či dostačující.

  %p
    Tento projekt má za cíl být
    = link_to "lepší konvencí pro changelog.", data.links.changelog
    Pochází z pozorování osvědčených postupů v open source komunitě a jejich
    shromažďování.

  %p
    Zdravá kritika, diskuse a návrhy na zlepšení
    = link_to "jsou vítány.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Jak by se soubor s changelogem měl jmenovat?

  %p
    Pojmenujte ho <code>CHANGELOG.md</code>. Některé projekty
    používají <code>HISTORY</code>, <code>NEWS</code> nebo <code>RELEASES</code>.

  %p
    Zatímco je snadné si myslet, že na názvu souboru vašeho changelogu
    až tak nezáleží, proč koncovým uživatelům ztěžovat hledání významných
    změn?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    A co GitHub Releases?

  %p
    Je to skvělá iniciativa. Služba #{link_to "Releases", data.links.github_releases} může být
    použita na proměnu obyčejných git tagů (například tag s názvem
    <code>v1.0.0</code>) na bohaté poznámky k vydání manuálním přidáním
    daných poznámek, nebo může pullnout zprávy z anotovaných git tagů
    a udělat poznámky k vydání z nich.

  %p
    GitHub Releases však vytvářejí nepřenosný changelog, který může být
    zobrazen uživatelům jen v kontextu GitHubu. Je možné je udělat
    velmi podobné formátu projektu Udržuj Changelog, ale to má
    tendenci být trochu náročnější.

  %p
    Aktuální verze GitHub Releases také pravděpodobně není příliš
    objevitelná koncovými uživateli, narozdíl od typických souborů
    s názvy psanými velkými písmeny (<code>README</code>,
    <code>CONTRIBUTING</code>, atd.). Další menší problém je, že
    Releases aktuálně nenabízí možnost odkazovat na commit logy mezi
    jednotlivými vydáními.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Mohou changelogy být automaticky parsovány?

  %p
    Je to složité, protože lidé používají mnoho rozdílných formátů a názvů
    souborů.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Ruby gem vytvořený týmem
    Gemnasium, který parsuje mnoho (ale ne všechny)
    changelogy open source projektů.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    A co zpětně stažená vydání?

  %p
    Stažená vydání jsou verze, které musely být zpětně odebrány kvůli vážné
    chybě nebo bezpečnostním rizikům. Tyto verze se často v changelogu ani
    neobjevují, ale měly by. Zobrazovat by se měly takto:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Tag <code>[YANKED]</code> je křiklavý naschvál. Je důležité, aby si ho
    lidé všimli. Díky tomu, že je v hranatých závorkách, je také jednodušší ho
    parsovat programem.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Měl by se changelog někdy přepisovat?

  %p
    Jistě. Vždy se najde dobrý důvod pro zlepšení changelogu. Sám často otevírám
    pull requesty pro přidání chybějících vydání v open source projektech s
    neudržovanými changelogy.

  %p
    Je také možné, že zjistíte, že v poznámkách k verzi ve vašem projektu není
    zmíněna změna, která něco rozbila. V tom případě je samozřejmě důležité,
    aby byl changelog aktualizován.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Jak mohu přispět?

  %p
    Tento dokument není čistá <strong>pravda</strong>; je to můj pečlivě
    zvážený názor, společně s informacemi a příklady, které jsem shromáždil.

  %p
    Je tomu tak proto, že chci aby naše komunita došla ke společné shodě.
    Věřím, že diskuse je stejně důležitá jako konečný výsledek.

  %p
    Takže prosím, <strong>#{link_to "přiložte ruku k dílu", data.links.repo}</strong>.

.press
  %h3 Rozhovory
  %p
    Zúčastnil jsem se #{link_to "podcastu The Changelog", data.links.thechangelog},
    abych promluvil o tom, proč by se správci projektů a přispěvatelé měli
    starat o changelogy a také o motivaci za tímto projektem.
```

## File: source/cs/1.1.0/index.html.haml
```haml
---
title: Udržuj Changelog
description: Nenech své kamarády sypat git logy do changelogů.
language: cs
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Verze
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Co je to changelog?

  %p
    Changelog je soubor, který obsahuje organizovaný, chronologicky seřazený
    seznam podstatných změn pro každou verzi daného projektu.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Proč udržovat changelog?

  %p
    Aby uživatelé a přispěvatelé přesně věděli, jaké podstatné změny byly
    provedeny mezi jednotlivými vydáními (verzemi) daného projektu.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kdo potřebuje changelog?

  %p
    Lidé. Ať už se jedná o spotřebitele nebo vývojáře, koncoví uživatelé
    softwaru jsou lidská stvoření, kterým záleží na tom, co software obsahuje.
    Když se daný software změní, lidé chtějí vědět proč a jak.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Jak vytvořit dobrý changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Hlavní zásady

  %ul
    %li
      Changelogy jsou <em>pro lidi</em>, ne pro stroje.
    %li
      Changelog by měl obsahovat záznam pro každou verzi.
    %li
      Stejné typy změn by měly být seskupené.
    %li
      Měla by existovat možnost odkazovat na jednotlivé verze a sekce.
    %li
      Poslední verze je na prvním místě.
    %li
      U každé verze je poznamenáno datum jejího vydání.
    %li
      Zmiňte, zda se držíte #{link_to "Sémantického verzování", data.links.semver}

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Typy změn

  %ul
    %li
      %code Added
      pro nové funkce.
    %li
      %code Changed
      pro změny v existující funkcionalitě.
    %li
      %code Deprecated
      pro funkce, které budou brzy odstraněny.
    %li
      %code Removed
      pro odstraněné funkce.
    %li
      %code Fixed
      pro opravy chyb.
    %li
      %code Security
      v případě bezpečnostních zranitelností.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Jak minimalizovat úsilí potřebné k udržování changelogu?

  %p
    Udržováním <code>Unreleased</code> sekce na začátku souboru pro zaznamenávání
    nadcházejících změn.

  %p To plní hned dva účely:

  %ul
    %li
      Lidé uvidí, jaké změny mohou očekávat v následujících vydáních
    %li
      V čas vydání stačí přesunout změny z <code>Unreleased</code> sekce
      do sekce nového vydání.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Mohou být changelogy špatné?

  %p Ano. Zde je několik případů, kdy mohou být opakem užitečného.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffy z commit logu

  %p
    Používání diffů z commit logu jako changelogu je špatný nápad:
    jsou plné šumu. Obsahují věci jako merge commity, commity s
    nejasnými nadpisy, změny v dokumentaci, atd.

  %p
    Účelem commitu je zdokumentovat krok v evoluci zdrojového kódu.
    Některé projekty commity pročišťují, jiné zas ne.

  %p
    Účelem záznamu v changelogu je zdokumentovat podstatné změny,
    často napříč několika commity, a jasně je sdělit koncovým uživatelům.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorování odstraněných funkcí
  %p
    Když lidé upgradují z jedné verze na druhou, mělo by jim být bolestně
    jasné, když se něco rozbije. Mělo by být možné nejprve upgradovat na verzi,
    která oznámí, jaké funkce budou odstraněny, dané funkce odstranit
    a poté upgradovat na verzi, kde jsou zmíněné funkce již odstraněny.

  %p
    Když už nic jiného, je dobré alespoň vypsat odstraněné funkce a
    změny, které něco rozbíjí, do changelogu.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Matoucí data

  %p
    Regionální formáty dat se liší napříč světem a je často složité
    najít formát, který je přátelský a intuitivní pro všechny.
    Výhodou dat formátovaných jako <code>2017-07-17</code> je pořadí
    jednotek od největší po nejmenší: rok, měsíc a den. Tento formát
    se navíc nepřekrývá s jinými, narozdíl od některých regionálních
    formátů, které prohazují pozici měsíce a dne. Díky těmto důvodům,
    a také faktu, že je tento formát #{link_to "ISO standard", data.links.isodate},
    je doporučeným formátem pro záznamy v changelogu.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Nekonzistentní změny

  %p
    Changelog, který uvádí pouze některé změny, může být stejně nebezpečný
    jako absence changelogu. I když mnoho změn nemusí být relevantních -
    například odstranění jednoho prázdného znaku nemusí být nutně
    uvedeno – všechny důležité změny by měly být uvedeny v changelogu.
    Nedůsledným uplatňováním změn si mohou vaši uživatelé mylně myslet,
    že changelog je jediný správný zdroj. Mělo by tak být. S velkou mocí 
    přichází velká zodpovědnost - mít dobrý changelog znamená mít ho neustále
    aktualizovaný.

  %aside
    Je toho však víc. Pomozte mi sesbírat tyto antipatterny
    = link_to "otevřením issue", data.links.issue
    nebo pull requestu.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Často kladené otázky

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Existuje pro formát changelogu nějaký standard?

  %p
    Ne. Je tu GNU stylová příručka pro changelog, nebo ta dvouodstavcová
    GNU "směrnice" pro NEWS soubor. Ani jedno však není vhodné či dostačující.

  %p
    Tento projekt má za cíl být
    = link_to "lepší konvencí pro changelog.", data.links.changelog
    Pochází z pozorování osvědčených postupů v open source komunitě a jejich
    shromažďování.

  %p
    Zdravá kritika, diskuse a návrhy na zlepšení
    = link_to "jsou vítány.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Jak by se soubor s changelogem měl jmenovat?

  %p
    Pojmenujte ho <code>CHANGELOG.md</code>. Některé projekty
    používají <code>HISTORY</code>, <code>NEWS</code> nebo <code>RELEASES</code>.

  %p
    Zatímco je snadné si myslet, že na názvu souboru vašeho changelogu
    až tak nezáleží, proč koncovým uživatelům ztěžovat hledání významných
    změn?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    A co GitHub Releases?

  %p
    Je to skvělá iniciativa. Služba #{link_to "Releases", data.links.github_releases} může být
    použita na proměnu obyčejných git tagů (například tag s názvem
    <code>v1.0.0</code>) na bohaté poznámky k vydání manuálním přidáním
    daných poznámek, nebo může pullnout zprávy z anotovaných git tagů
    a udělat poznámky k vydání z nich.

  %p
    GitHub Releases však vytvářejí nepřenosný changelog, který může být
    zobrazen uživatelům jen v kontextu GitHubu. Je možné je udělat
    velmi podobné formátu projektu Udržuj Changelog, ale to má
    tendenci být trochu náročnější.

  %p
    Aktuální verze GitHub Releases také pravděpodobně není příliš
    objevitelná koncovými uživateli, narozdíl od typických souborů
    s názvy psanými velkými písmeny (<code>README</code>,
    <code>CONTRIBUTING</code>, atd.). Další menší problém je, že
    Releases aktuálně nenabízí možnost odkazovat na commit logy mezi
    jednotlivými vydáními.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Mohou changelogy být automaticky parsovány?

  %p
    Je to složité, protože lidé používají mnoho rozdílných formátů a názvů
    souborů.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Ruby gem vytvořený týmem
    Gemnasium, který parsuje mnoho (ale ne všechny)
    changelogy open source projektů.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    A co zpětně stažená vydání?

  %p
    Stažená vydání jsou verze, které musely být zpětně odebrány kvůli vážné
    chybě nebo bezpečnostním rizikům. Tyto verze se často v changelogu ani
    neobjevují, ale měly by. Zobrazovat by se měly takto:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Tag <code>[YANKED]</code> je křiklavý naschvál. Je důležité, aby si ho
    lidé všimli. Díky tomu, že je v hranatých závorkách, je také jednodušší ho
    parsovat programem.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Měl by se changelog někdy přepisovat?

  %p
    Jistě. Vždy se najde dobrý důvod pro zlepšení changelogu. Sám často otevírám
    pull requesty pro přidání chybějících vydání v open source projektech s
    neudržovanými changelogy.

  %p
    Je také možné, že zjistíte, že v poznámkách k verzi ve vašem projektu není
    zmíněna změna, která něco rozbila. V tom případě je samozřejmě důležité,
    aby byl changelog aktualizován.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Jak mohu přispět?

  %p
    Tento dokument není čistá <strong>pravda</strong>; je to můj pečlivě
    zvážený názor, společně s informacemi a příklady, které jsem shromáždil.

  %p
    Je tomu tak proto, že chci aby naše komunita došla ke společné shodě.
    Věřím, že diskuse je stejně důležitá jako konečný výsledek.

  %p
    Takže prosím, <strong>#{link_to "přiložte ruku k dílu", data.links.repo}</strong>.

.press
  %h3 Rozhovory
  %p
    Zúčastnil jsem se #{link_to "podcastu The Changelog", data.links.thechangelog},
    abych promluvil o tom, proč by se správci projektů a přispěvatelé měli
    starat o changelogy a také o motivaci za tímto projektem.
```

## File: source/da/1.1.0/index.html.haml
```haml
---
title: Hold en changelog
description: Du skal ikke lade dine venner smide git logs i changelogs.
language: da
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Hvad er en changelog?

  %p
    En changelog er en fil, der indeholder en organiseret, kronologisk
    sorteret liste af bemærkelsesværdige ændringer for hver version
    af et projekt.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Hvorfor have en changelog?

  %p
    For at gøre det nemmere for brugere og bidragsydere at se præcist
    hvilke bemærkelsesværdige ændringer, der er lavet mellem hver
    udgivelse eller version af et projekt.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Hvem skal bruge en changelog?

  %p
    Det skal folk. Om det er forbrugere eller udviklere, er slutbrugere
    mennesker, der bekymrer sig om, hvad der er i et stykke software. Når
    softwaren ændrer sig, vil folk gerne vide hvad, og hvordan.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Hvordan laver jeg en god changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Retningslinjer

  %ul
    %li
      Changelogs er til <em>for mennesker</em>, ikke maskiner.
    %li
      Der skal være et indlæg for hver eneste version.
    %li
      Ens typer af ændringer skal grupperes.
    %li
      Versioner og sektioner skal være link-bare.
    %li
      Den seneste version står først.
    %li
      Udgivelsesdatoen for hver version vises.
    %li
      Nævn hvorvidt du følger #{link_to "Semantisk Versionering", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Ændringstyper

  %ul
    %li
      %code Tilføjet (Added)
      til nye funktioner.
    %li
      %code Ændret (Dhanged)
      til rettelser i eksisterende funktionalitet.
    %li
      %code Udfaset (Deprecated)
      til funktioner, der fjernes snart.
    %li
      %code Fjernet (Removed)
      til fjernede funktioner.
    %li
      %code Rettet (Fixed)
      til fejlrettelser.
    %li
      %code Sikkerhed (Security)
      i tilfælde af sårbarheder.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Hvordan kan jeg reducere indsatsen ved at opretholde en changelog?

  %p
    Hav en <code>Unreleased</code> sektion i toppen til kommende
    rettelser.

  %p Dette tjener to formål:

  %ul
    %li
      Folk kan se, hvilke rettelser de kan forvente i den kommende
      udgivelse
    %li
      Ved udgivelsen kan du flytte <code>Unreleased</code> sektionens
      rettelser ind i en ny versionssektion.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kan changelogs være dårlige?

  %p Ja. Her er et par måder, de kan være mindre brugbare.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    At bruge commit log differenser som changelog er en dårlig idé:
    De er fyldt med støj. Ting som merge-commits, commits med obskure titler,
    dokumentationsrettelser osv.

  %p
    Formålet med et commit er at dokumentere et trin i udviklingen af
    kildekoden. Nogle projekter rydder op i deres commits, andre gør
    ikke.

  %p
    Formålet med en changelog-indlæg er at dokumentere de bemærkelsesværdige
    forskelle, ofte fordelt over flere commits, for at formidle dem klart
    til slutbrugerene.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    At ignorere udfasninger

  %p
    Når folk opgraderer fra én version til en anden, skal det være helt
    tydeligt, hvorvidt noget vil gå i stykker. Det skal være muligt
    at opgradere til en version, der opsummerer udfasninger, fjerner
    gamle udfasniner, og opgradere til dén version, hvor udfasningerne
    bliver til oprydninger.

  %p
    Hvis du ikke gør andet, så notér udfasninger, fjernede funktioner
    og andre ødelæggende ændringer.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Forvirrende datoer

  %p
    Regionale dataformater er varierende i hele verden, og det er ofte
    besværligt at finde et læsbart datoformat, der føles intuitivt for alle
    Fordelen ved datoer formatteret som <code>2017-07-17</code> er at de
    følger rækkefølgen for største til mindste enhed: år, måned og dag.
    Dette format overlapper heller ikke på en tvetydig måde med andre
    datoformater, der ombytter måneden og daten. Af disse årsager, og
    det faktum at dette datoformat er en #{link_to "ISO standard", data.links.isodate},
    er dette det anbefalede datoformat for changelog-indlæg.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Ukonsistente rettelser

  %p
    En changelog, der kun nævner nogle af ændringerne, kan være lige så
    farlig som ikke at have en changelog. Mens mange af ændringerne
    måske ikke er relevante - for eksempel er det ikke være nødvendigt at
    notere, at vi har fjernet et enkelt mellemrum - bør væsentlige
    ændringer være nævnt i changeloggen. Ved inkonsekvent at anvende
    ændringer, kan dine brugere fejlagtigt tro, at changelog er den
    eneste kilde til sandhed. Det bør den være. Med store
    magtbeføjelser følger store forpligtigelser. At have en god changelog
    er at have en konsekvent opdateret changelog.

  %aside
    Der er mere. Hjælp mig med at samle disse fælder ved at
    = link_to "åbne et issue", data.links.issue
    eller et pull-request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Ofte stillede spørgsmål

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Er der et standard changelog-format?

  %p
    Ikke rigtig. Der er #{link_to "GNU changelog style guide", data.links.gnustyle},
    eller #{link_to "two-paragraph-long GNU NEWS file", data.links.gnunews}. De
    er begge utilstrækkelige eller uhensigtsmæssige.

  %p
    Dette projekt sigter efter at være
    = link_to "en bedre changelog konvention.", data.links.changelog
    Det kommer af at observere god praksis i opensource-verdenen
    og samle disse.

  %p
    Konstruktiv kritik diskussion og forbedringsforslag
    = link_to "er velkomne.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Hvad skal changelog-filen hedde?

  %p
    Kald den <code>CHANGELOG.md</code>. Nogle projekter bruger
    <code>HISTORY</code>, <code>NEWS</code> eller <code>RELEASES</code>.

  %p
    Selvom det er nemt at mene, at navnet på changelog-filen er
    irrelevant, hvorfor så gøre det sværere for dine slugbrugere
    at finde bemærkelsesværdige ændringer?


  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Hvad med GitHub Releases?

  %p
    Det er et godt initiativ. #{link_to "Releases", data.links.github_releases} kan bruges
    til at gøre simple git tags (for eksempel, et tag <code>v1.0.0</code>)
    til fyldige udgivelsesnoter ved manuelt at tilføje noter til disse
    eller bruge en annoteret git-tag-besked som udgivelsesnoter.

  %p
    GitHub Releases laver en ikke-flytbar changelog, der kun kan vises
    til bruger i konteksten af GitHub. Det er muligt at få dem til at
    ligne Hold en changelog-formatet, men det har en tendens til at
    være lidt mere kompliceret.

  %p
    Den nuværende version af GitHub releases er, diskutérbart,
    ikke særligt synlig for slutbrugere, ikke lig de typiske
    kapitaliserede filer (<code>README</code>,
    <code>CONTRIBUTING</code>, osv.). Et andet mindre problem, er
    at interfacet ikke tilbyder links til commit logs mellem hvert
    release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Kan changelogs læses og fortolkes automatisk?

  %p
    Det er besværligt, fordi folk følger vidt forskellige formater
    og filnavne.

  %p
    #{link_to "Vandamme", data.links.vandamme} er en Ruby gem lavet af
    holdet bag Gemnasium, som læser og fortolker mange (men ikke
    alle) open source projekters changelogs.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Hvad med tilbagetrukkede udgivelser?

  %p
    Tilbagetrukkede udgivelser er versioner, der er blevet trukket
    tilbage på grund af en seriøs fejl eller sikkerhedsbrist. Ofte
    vises disse versioner slet ikke i changelogs. Det bør de. Dette
    er hvordan du bør vise dem:

  %p <code>## 0.0.5 - 2014-12-13 [YANKED]</code>

  %p
    Notationen <code>[YANKED]</code>, er højlydt af en årsag: Det er
    vigtig information for folk. Siden den er pakket ind i klammer, er
    den nemmere at fortolke og læse systematiseret.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Bør man nogensinde omskrive en changelog?

  %p
    Klart! Der er altid gode grunde til at forbedre en changelog.
    Jeg åbner jævnligt pull-requests, der tilføjer manglende releases
    til open source projekter uden en vedligeholdt changelog.

  %p
    Det er også muligt, at du vil opdage at du har glemt at addressere
    en ødelæggende rettelse i noterne for en version. Det er selvfølgelig
    vigtigt at du opdaterer en changelog i disse tilfælde.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Hvordan kan jeg hjælpe til?

  %p
    Dette dokument er ikke <strong>sandheden</strong>; det er min
    velovervejede mening sammen med information og eksempler jeg har
    samlet.

  %p
    Det er fordi, jeg gerne vil have at vores fællesskab når en konsensus.
    Jeg mener at diskussionen er lige så vigtig som slutresultatet.
  %p
    Så, <strong>#{link_to "vær med!", data.links.repo}</strong>

.press
  %h3 Samtaler
  %p
    Jeg var med i #{link_to "The Changelog podcast", data.links.thechangelog} for
    at snakke om, hvorfor vedligeholdere og bidragsydere bør bekymre sig
    om changelogs og også om motivationen bag dette projekt.
```

## File: source/de/0.3.0/index.html.haml
```haml
---
title: Führe ein CHANGELOG
description: Lass Deine Freunde nicht CHANGELOGs mit git logs füllen
language: de
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Was ist ein Changelog?
  Ein Changelog ist eine Datei, welche eine handgepflegte, chronologisch sortierte
  Liste aller erheblichen Änderungen enthält, die zwischen Veröffentlichungen (oder Versionen)
  des Projekts umgesetzt wurden.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Was ist der Zweck eines Changelogs?
  Es Benutzern und Entwicklern einfacher zu machen, gerade die beachtenswerten Änderungen, welche
  zwischen Veröffentlichungen (oder Versionen) des Projekts gemacht wurden, zu sehen.

  ### Warum sollte mich das kümmern?
  Weil Software-Werkzeuge für Menschen gemacht sind. Wenn es Dich nicht kümmert, warum
  trägst Du dann zu Open Source bei? Wenn Du tief in Dich gehst, findest Du bestimmt
  einen kleinen Teil, dem das wichtig ist.

  Ich [habe mit Adam Stacoviak and Jerod Santo im The Changelog-Podcast][thechangelog] (passt, oder?)
  darüber gesprochen (Englisch), weshalb sich Entwickler darum kümmern sollten und über die
  Beweggründe hinter diesem Projekt. Falls Du die Zeit hast (1:06), es lohnt sich, reinzuhören.

  ### Was macht ein gutes Changelog aus?
  Schön, dass Du fragst.

  Ein gutes Changelog hält sich an die folgenden Prinzipien:

  - Es ist für Menschen, nicht Maschinen, gemacht, deshalb ist Lesbarkeit entscheidend.
  - Es ist einfach, jeden Bereich zu verlinken (also besser Markdown als einfacher Text).
  - Ein Unterkapitel pro Version.
  - Liste die Releases in umgekehrt chronologischer Reihenfolge auf (neuestes zuoberst).
  - Schreib alle Daten im Format `JJJJ-MM-TT`. (Beispiel: `2012-06-02` für den `2. Juni 2012`.) Es ist international, [vernünftig](https://xkcd.com/1179/), und sprachunabhängig.
  - Erwähne explizit, ob das Projekt nach [Semantic Versioning][semver] geführt wird.
  - Jede Version sollte:
    - Das Release-Datum im oben genannten Format auflisten.
    - Änderungen wie folgt gruppieren, um ihren Einfluss auf das Projekt zu beschreiben:
      - `Added` für neue Features.
      - `Changed` für Änderungen an bestehender Funktionalität.
      - `Deprecated` für einst stabile Features, welche in zukünftigen Versionen entfernt werden.
      - `Removed` für Deprecated-Features, welche in dieser Version entfernt wurden.
      - `Fixed` für Bug-Fixes.
      - `Security` um Benutzer im Fall von Sicherheitsrisiken zu einer Aktualisierung aufzufordern.

  ### Wie kann ich den Aufwand so klein wie möglich halten?
  Hab immer einen `"Unreleased"`-Abschnitt zuoberst, um Änderungen im Auge zu behalten.

  Dies verfolgt zwei Ziele:

  - Man kann sehen, welche Änderungen in den nächsten Releases zu erwarten sind
  - Wenn es Zeit für den Release ist, kannst Du einfach `"Unreleased"` auf die
    Versionsnummer ändern und einen neuen `"Unreleased"`-Titel oben einfügen.

  ### Was bringt Einhörner zum weinen?
  Also… Schauen wir uns das an.

  - **Einen Diff von Commit-Logs ausgeben.** Mach das nicht, das hilft niemandem.
  - **Nicht mehr unterstützte Funktionen nicht hervorzuheben.** Wenn man von einer auf
    eine andere Version aktualisiert, sollte schmerzhaft klar sein, wenn dadurch etwas
    nicht mehr funktioniert.
  - **Datum in regionalen Formaten.** In den USA schreibt man den Monat zuerst
    ("06-02-2012" für den 2. Juni 2012, was *keinen* Sinn macht), während im Rest
    der Welt häufig ein roboterhaft aussehendes "2 June 2012" geschrieben, jedoch
    völlig anders ausgesprochen wird. "2012-06-02" funktioniert logisch vom grössten
    zum kleinsten Wert, überlagert sich nicht auf mehrdeutige Art mit anderen Datumsformaten
    und ist ein [ISO-Standard](http://www.iso.org/iso/home/standards/iso8601.htm). Deshalb
    ist es das empfohlene Datumsformat für Changelogs

  Das war noch nicht alles. Hilf mir, diese Einhorn-Tränen zu sammeln und [eröffne ein Issue][issues]
  oder einen Pull-Request.

  ### Gibt es ein standardisiertes Changelog-Format?
  Leider nicht. Beruhige Dich. Ich weiss, dass Du wie wild nach dem Link für den
  GNU Changelog Styleguide oder den zwei Absätze langen GNU NEWS-Datei "Leitfaden"
  suchst. Der GNU Styleguide ist ein guter Anfang, aber er ist leider sehr naiv.
  Es ist sicher nichts falsch daran, naiv zu zu sein, aber beim Verfassen eines Leitfadens
  ist es nicht wirklich hilfreich. Vor allem nicht, wenn es viele Spezialfälle zu beachten gibt.

  Dieses Projekt [enthält das, wovon ich hoffe, dass es zu einer besseren CHANGELOG-Datei-Konvention][CHANGELOG]
  wird. Ich glaube nicht, dass der status quo gut genug ist, und ich denke, dass wir als Community
  eine bessere Konvention entwickeln können, wenn wir Bewährtes aus echten
  Software-Projekten entnehmen. Schau Dich um und denk daran, dass
  [Diskussionen und Verbesserungsvorschläge sehr willkommen sind][issues]!

  ### Wie soll ich meine Changelog-Datei nennen?
  Nun, falls Du es noch nicht am Beispiel oben gesehen hast, `CHANGELOG.md`
  ist bisher die beste Konvention.

  Einige Projekte nutzen auch `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`, etc.

  Es ist ein Chaos. Alle diese Namen machen es nur schwerer, die Datei zu finden.

  ### Wieso sollte man nicht einfach ein `git log` Diff verwenden?
  Weil log Diffs voller unnötiger Information sind - von Natur aus. Sie wären nicht
  einmal ein geeignetes Changelog in einem hypothetischen Projekt, welches von perfekten
  Menschen geführt wird, welche sich niemals vertippen, niemals vergessen, neue Dateien
  zu comitten und nie einen Teil eines Refactorings übersehen.
  Der Zweck eines Commits ist es, einen atomaren Schritt eines Prozesses zu dokumentieren,
  welcher den Code von einem Zustand in den nächsten bringt. Der Zweck eines Changelogs
  ist es, die nennenswerten Veränderungen zwischen diesen Zuständen zu dokumentieren.

  Der Unterschied zwischen dem Changelog und dem Commit-Log ist wie der Unterschied
  zwischen guten Kommentaren und dem Code selbst:
  Das eine beschreibt das *wieso*, das andere das *wie*.

  ### Kann man Changelogs automatisiert parsen?
  Es ist nicht einfach, weil äusserst unterschiedliche Formate und Dateinamen verwendet
  werden.

  [Vandamme][vandamme] ist ein Ruby gem vom [Gemnasium][gemnasium]-Team, welches
  viele (aber nicht alle) Changelogs von Open-Source-Projekten parsen kann.

  ### Wieso schreibst Du mal "CHANGELOG" und mal "Changelog"?
  "CHANGELOG" ist der Name der Datei selbst. Es ist ein bisschen laut, aber
  es ist eine historische Konvention, welche von vielen Open-Source-Projekten
  angewendet wird. Andere Beispiele von ähnlichen Dateien sind [`README`][README],
  [`LICENSE`][LICENSE] und [`CONTRIBUTING`][CONTRIBUTING].

  Die Grossschreibung (welche in alten Betriebssystemen dafür gesorgt hat,
  dass die Dateien zuerst aufgelistet wurden) wird verwendet, um die Aufmerksamkeit
  auf diese Dateien zu lenken. Da sie wichtige Metadaten über das Projekt enthalten,
  können sie wichtig für jeden sein, der das Projekt gerne benutzen oder mitentwickeln
  möchte, ähnlich wie [Open-Source-Projekt-Badges][shields].

  Wenn ich über ein "Changelog" spreche, dann meine ich die Funktion der Datei:
  das Festhalten von Änderungen.

  ### Wie sieht es mit zurückgezogenen Releases aus?
  Sogenannte "Yanked Releases" sind Versionen, welche wegen schwerwiegenden
  Bugs oder Sicherheitsproblemen zurückgezogen werden mussten. Häufig kommen
  diese im Changelog gar nicht vor. Das sollten sie aber. So solltest Du diese
  darstellen:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Das `[YANKED]`-Tag ist aus einem guten Grund laut. Es ist wichtig, dass es
  wahrgenommen wird. Dass es von Klammern umfasst ist, vereinfacht auch
  das automatisierte Parsen.

  ### Sollte ich ein Changelog je umschreiben?
  Klar. Es gibt immer gute Gründe, ein Changelog zu verbessern. Ich öffne
  regelmässig Pull Requests um Open-Source-Projekten mit schlecht gewarteten
  Changelogs fehlende Releases hinzuzufügen.

  Es ist auch möglich, dass Du eine wichtige Änderung vergessen hast, in einer
  Version zu erwähnen. Natürlich ist es in diesem Fall wichtig, das Changelog
  zu aktualisieren.

  ### Wie kann ich mithelfen?
  Dieses Dokument ist nicht die **Wahrheit**; Es ist meine wohl überlegte Meinung,
  zusammen mit von mir zusammengetragenen Informationen und Beispielen.
  Obwohl ich im [GitHub-Repository][gh] ein [CHANGELOG][] führe, habe ich
  keine echten *Releases* oder klare zu befolgenden Regeln geschrieben (wie dies
  zum Beispiel [SemVer.org][semver] tut).

  Das ist so, weil ich möchte, dass die Community sich einig wird. Ich glaube,
  dass die Diskussion genau so wichtig wie das Endresultat ist.

  Deshalb [**pack bitte mit an**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/de/1.0.0/index.html.haml
```haml
---
title: Führe ein CHANGELOG
description: Lass Deine Freunde nicht CHANGELOGs mit git logs füllen.
language: de
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Was ist ein Changelog?

  %p
    Ein Changelog ist eine Datei, die eine handgepflegte, chronologisch sortierte
    Liste aller erheblichen Änderungen enthält, die zwischen einzelnen Veröffentlichungen
    (oder Versionen) des Projekts umgesetzt wurden.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Was ist der Zweck eines Changelogs?

  %p
    Ein Changelog soll es  Benutzern und Entwicklern einfacher machen, gerade die
    beachtenswerten Änderungen, die zwischen Veröffentlichungen (oder Versionen)
    des Projekts gemacht wurden, zu sehen.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Wer braucht schon ein Changelog?

  %p
    Menschen brauchen es. Seien es Konsumenten oder Entwickler, die Endnutzer der Software
    sind Menschen, die es interessiert, was in der Software passiert. Wenn sich die Software ändert,
    dann wollen diese Menschen wissen, wie und warum sie sich ändert.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Wie erstelle ich einen guten Changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Grundlegende Prinzipien

  %ul
    %li
      Changelogs werden <em>für Menschen</em> geschrieben, nicht für Maschinen.
    %li
      Für jede einzelne Version sollte es einen Eintrag geben.
    %li
      Änderungen der selben Art sollten in Bereiche gruppiert werden.
    %li
      Versionen und Bereiche sollten verlinkt werden können.
    %li
      Die neuste Version kommt als erstes.
    %li
      Das Release-Datum jeder Version muss angegeben sein.
    %li
      Gib an, ob du dich an die #{link_to "Semantische Versionierung", data.links.semver} hältst.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Arten von Änderungen

  %ul
    %li
      %code Added
      für neue Features.
    %li
      %code Changed
      für Änderungen an der bestehenden Funktionalität.
    %li
      %code Deprecated
      für Features, die in zukünftigen Versionen entfernt werden.
    %li
      %code Removed
      für Deprecated-Features, die in dieser Version entfernt wurden.
    %li
      %code Fixed
      für alle Bug-Fixes.
    %li
      %code Security
      um Benutzer im Fall von geschlossenen Sicherheitslücken zu einer Aktualisierung aufzufordern.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Wie kann ich den Aufwand der Changelog-Pflege so klein wie möglich halten?

  %p
    Habe immer einen <code>Unreleased</code>-Abschnitt über der neusten Version,
    um zukünftige Änderungen im Auge zu behalten.

  %p Dies hat zwei Vorteile:

  %ul
    %li
      Menschen können sehen, welche Änderungen sie mit dem nächsten Release zu erwarten haben.
    %li
      Wenn der Zeitpunkt für den Release gekommen ist, kannst du den Inhalt des
      <code>Unreleased</code>-Abschnitts in den neuen Release-Abschnitt verschieben.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kann man beim Changelog etwas falsch machen?

  %p Ja. Hier sind einige Dinge, die eher unbrauchbar sind.


  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Einen Diff von Commit-Logs ausgeben

  %p
    Commit-Logs in einem Changelog sind eine schlechte Idee: Sie beinhalten lauter
    überflüssige Dinge, wie Merge-Commit, Commits mit schlechten Bezeichnungen,
    Änderungen an der Dokumentation, etc.

  %p
    Der Sinn eines Commits ist die Entwicklung des Source Code zu dokumentieren.
    Manche Projekte haben saubere Commits, andere nicht.

  %p
    Der Sinn eines Changelog-Eintrags ist die Dokumentation der merkbaren
    Unterschiede, die meist über mehrere Commits hinweg entstanden sind, dem
    Endnutzer klar und deutlich zu kommunizieren.


  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Features ohne Deprecation-Warnung entfernen

  %p
    Wenn der Endnutzer auf eine neue Version upgradet, sollte ihm unmissverständlich
    klar gemacht werden, wenn etwas kaputt gehen wird. Es sollte immer möglich sein,
    zu einer Version zu upgraden, die die zu entfernenden Features auflistet, um so
    in seinem Source Code auf diese Features zu verzichten. Anschließend sollte man
    auf eine Version upgraden können, in der die Features entfernt wurden.

  %p
    Auch wenn du sonst nichts geändert hast, liste trotzdem alle veralteten und
    entfernten Features, sowie jede funktionsgefährdende Änderung in deinem Changelog auf.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Verwirrende Datumsformate

  %p
    In den USA setzen die Menschen den Monat an den Anfang eines Datums
    (<code>06-02-2012</code> für den 2. Juni 2012), wohingegen viele Menschen
    im Rest der Welt ein roboterhaft aussehendes <code>2 June 2012</code>
    schreiben, aber es völlig unterschiedlich aussprechen. <code>2012-06-02</code>
    folgt der Logik vom größten zum kleinsten Wert, kann nicht mit anderen Formaten
    verwechselt werden und ist ein #{link_to "ISO Standard", data.links.isodate}. Deshalb
    ist es das empfohlene Datumsformat in Changelogs.

  %aside
    Es gibt sicher noch mehr. Hilf mir alle schlechten Tipps zu sammeln, indem du
    = link_to "ein Issue eröffnest", data.links.issue
    oder einen Pull-Request erstellst.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Häufig gestellte Fragen

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Gibt es ein standardisiertes Changelog-Format?

  %p
    Leider nicht. Es gibt zwar den GNU Changelog Styleguide oder den
    zwei Absätze langen GNU NEWS-Datei "Leitfaden". Beide sind aber
    unzureichend.

  %p
    Dieses Projekt versucht
    = link_to "eine bessere Changelog Konvention zu sein.", data.links.changelog
    Dazu beobachten wir bewährte Praktiken aus der Open Source Community
    und tragen sie zusammen.

  %p
    Gesunde Kritik, Diskussionen und Verbesserungsvorschläge
    = link_to "sind herzlich willkommen.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Wie sollte die Changelog-Datei benannt sein?

  %p
    Nenne sie <code>CHANGELOG.md</code>. Manche Projekte verwenden auch
    <code>HISTORY</code>, <code>NEWS</code> oder <code>RELEASES</code>.

  %p
    Man könnte zwar meinen, dass der Name der Changelog-Datei keine große
    Bedeutung hat, aber warum sollte man es den Endnutzern nicht einfach machen,
    die Änderungen des Projekts zu finden, indem man einen einheitlichen Namen
    verwendet?


  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Was ist mit GitHub Releases?

  %p
    Prinzipiell sind #{link_to "GitHub Releases", data.links.github_releases} eine gute Sache.
    Sie können dazu benutzt werden, um einfache Git Tags (zum Beispiel einen
    Tag namens <code>v1.0.0</code>) mit vielen Hinweisen zum Release
    auszustatten, indem man sie manuell bearbeitet, oder die Änderungen in eine
    Git Tag Nachricht schreibt, wo sie durch GitHub automatisch in die
    Release-Notizen gesetzt werden.

  %p
    Leider lassen sich GitHub Releases aber nicht so einfach exportieren
    und sind nur über GitHub selber zu lesen. Man kann sie auch so gestalten,
    dass sie dem Keep a Changelog Format sehr ähnlich sehen, aber dazu betreibt
    man in der Regel einen größeren Aufwand.

  %p
    Die derzeitige Version der GitHub Releases sind außerdem nicht leicht
    durch die Endnutzer zu finden, ganz im Gegenteil zu den typischerweise
    großgeschriebenen Dateien (<code>README</code>, <code>CONTRIBUTING</code>, etc.).
    Ein weiterer kleiner Nachteil ist, dass das Web Interface zurzeit keinen Link
    anbietet, um die Commits zwischen einzelnen Releases miteinander zu vergleichen.


  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Können Changelogs automatisiert ausgelesen werden?

  %p
    Es ist schwierig, weil Menschen meist unterschiedliche Formate und
    Dateinamen verwenden.

  %p
    #{link_to "Vandamme", data.links.vandamme} ist ein Ruby gem erstellt vom
    Gemnasium Team, das viele (aber nicht alle)
    Changelogs von Open-Source-Projekten einlesen kann.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Wie sieht es mit zurückgezogenen Releases aus?

  %p
    Sogenannte "Yanked Releases" sind Versionen, welche wegen schwerwiegenden
    Bugs oder Sicherheitsproblemen zurückgezogen werden mussten. Häufig kommen
    diese im Changelog gar nicht vor, aber das sollten sie. So solltest Du diese
    Versionen darstellen:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Der <code>[YANKED]</code> ist nicht ohne Grund großgeschrieben. Es ist wichtig,
    dass sie von Menschen bemerkt werden. Weil er von eckigen Klammern umgeben ist,
    kann man sie außerdem einfacher automatisiert einlesen.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Sollte ich ein Changelog je umschreiben?

  %p
    Klar. Es gibt immer gute Gründe, ein Changelog zu verbessern. Ich öffne
    regelmässig Pull Requests, um Open-Source-Projekten mit schlecht gewarteten
    Changelogs fehlende Releases hinzuzufügen.

  %p
    Es ist auch möglich, dass Du eine wichtige Änderung vergessen hast, in einer
    Version zu erwähnen. Natürlich ist es in diesem Fall wichtig, das Changelog
    zu aktualisieren.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Wie kann ich mithelfen?

  %p
    Dieses Dokument ist nicht die <strong>Wahrheit</strong>. Es ist meine wohl überlegte Meinung,
    zusammen mit von mir zusammengetragenen Informationen und Beispielen.

  %p
    So möchte ich erreichen, dass die Community einen Konsens findet. Ich glaube, dass
    die Disskussion genauso wichtig ist wie das Endergebnis.

  %p
    Also bitte <strong>#{link_to "bring dich ein", data.links.repo}</strong>.

.press
  %h3 Gespräche
  %p
    Ich habe im #{link_to "The Changelog podcast", data.links.thechangelog} darüber gesprochen,
    warum sich Entwickler und Mitwirkende eines Projekts um ein Changelog kümmern sollten,
    sowie meine Motivationen hinter diesem Projekt erklärt.
```

## File: source/de/1.1.0/index.html.haml
```haml
---
title: Führe ein CHANGELOG
description: Lass Deine Freunde nicht CHANGELOGs mit git logs füllen.
language: de
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Was ist ein Changelog?

  %p
    Ein Changelog ist eine Datei, die eine handgepflegte, chronologisch sortierte
    Liste aller erheblichen Änderungen enthält, die zwischen einzelnen Veröffentlichungen
    (oder Versionen) des Projekts umgesetzt wurden.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Was ist der Zweck eines Changelogs?

  %p
    Ein Changelog soll es  Benutzern und Entwicklern einfacher machen, gerade die
    beachtenswerten Änderungen, die zwischen Veröffentlichungen (oder Versionen)
    des Projekts gemacht wurden, zu sehen.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Wer braucht schon ein Changelog?

  %p
    Menschen brauchen es. Seien es Konsumenten oder Entwickler, die Endnutzer der Software
    sind Menschen, die es interessiert, was in der Software passiert. Wenn sich die Software ändert,
    dann wollen diese Menschen wissen, wie und warum sie sich ändert.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Wie erstelle ich einen guten Changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Grundlegende Prinzipien

  %ul
    %li
      Changelogs werden <em>für Menschen</em> geschrieben, nicht für Maschinen.
    %li
      Für jede einzelne Version sollte es einen Eintrag geben.
    %li
      Änderungen der selben Art sollten in Bereiche gruppiert werden.
    %li
      Versionen und Bereiche sollten verlinkt werden können.
    %li
      Die neuste Version kommt als erstes.
    %li
      Das Release-Datum jeder Version muss angegeben sein.
    %li
      Gib an, ob du dich an die #{link_to "Semantische Versionierung", data.links.semver} hältst.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Arten von Änderungen

  %ul
    %li
      %code Added
      für neue Features.
    %li
      %code Changed
      für Änderungen an der bestehenden Funktionalität.
    %li
      %code Deprecated
      für Features, die in zukünftigen Versionen entfernt werden.
    %li
      %code Removed
      für Deprecated-Features, die in dieser Version entfernt wurden.
    %li
      %code Fixed
      für alle Bug-Fixes.
    %li
      %code Security
      um Benutzer im Fall von geschlossenen Sicherheitslücken zu einer Aktualisierung aufzufordern.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Wie kann ich den Aufwand der Changelog-Pflege so klein wie möglich halten?

  %p
    Habe immer einen <code>Unreleased</code>-Abschnitt über der neusten Version,
    um zukünftige Änderungen im Auge zu behalten.

  %p Dies hat zwei Vorteile:

  %ul
    %li
      Menschen können sehen, welche Änderungen sie mit dem nächsten Release zu erwarten haben.
    %li
      Wenn der Zeitpunkt für den Release gekommen ist, kannst du den Inhalt des
      <code>Unreleased</code>-Abschnitts in den neuen Release-Abschnitt verschieben.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kann man beim Changelog etwas falsch machen?

  %p Ja. Hier sind einige Dinge, die eher unbrauchbar sind.


  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Einen Diff von Commit-Logs ausgeben

  %p
    Commit-Logs in einem Changelog sind eine schlechte Idee: Sie beinhalten lauter
    überflüssige Dinge, wie Merge-Commit, Commits mit schlechten Bezeichnungen,
    Änderungen an der Dokumentation, etc.

  %p
    Der Sinn eines Commits ist die Entwicklung des Source Code zu dokumentieren.
    Manche Projekte haben saubere Commits, andere nicht.

  %p
    Der Sinn eines Changelog-Eintrags ist die Dokumentation der erwähnenswerten
    Unterschiede, die meist über mehrere Commits hinweg entstanden sind, dem
    Endnutzer klar und deutlich zu kommunizieren.


  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Features ohne Deprecation-Warnung entfernen

  %p
    Wenn der Endnutzer auf eine neue Version upgradet, sollte ihm unmissverständlich
    klar gemacht werden, wenn etwas kaputt gehen wird. Es sollte immer möglich sein,
    zu einer Version zu upgraden, die die zu entfernenden Features auflistet, um so
    in seinem Source Code auf diese Features zu verzichten. Anschließend sollte man
    auf eine Version upgraden können, in der die Features entfernt wurden.

  %p
    Auch wenn du sonst nichts geändert hast, liste trotzdem alle veralteten und
    entfernten Features, sowie jede funktionsgefährdende Änderung in deinem Changelog auf.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Verwirrende Datumsformate

  %p
    Regionale Datumsformate variieren über den Erdball und es ist oft schwer
    ein menschenlesbares Datumsformat zu finden, das sich für jeden intuitiv anfühlt.
    Der Vorteil von Datumsformaten wie <code>2017-07-17</code> ist, dass sie von der
    größten zur kleinsten Zeiteinheit sortiert dargestellt werden: Jahre, Monate und Tage.
    Dieses Format überschneidet sich auch nicht uneindeutig mit anderen Datumsformaten,
    wie manche andere, regionale Formate die Stelle der Monate und Tage vertauschen.
    Diese Gründe, und der Umstand, dass dieses Datumsformat
    ein #{link_to "ISO Standard", data.links.isodate} ist, sind Argumente wieso dieses
    Format für Changelog-Einträge empfohlen ist.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Inkonsistente Änderungen
  %p
    Ein Changelog, das nur einen Teil der Änderung beinhaltet, kann genauso gefährlich sein,
    wie gar kein Changelog zu führen. Auch wenn manche Änderung nicht relevant sein mögen - zum
    Beispiel muss das Entfernen eines einzelnen Leerzeichens nicht in jedem Fall
    protokolliert werden - sollten alle wichtigen Änderung im Changelog erwähnt werden.
    Werden Änderungen nur inkonsistent aufgezeichnet, könnten Benutzer fälschlicherweise
    zu dem Schluss kommen, dass das Changelog die einzige Quelle der Wahrheit sei. Das sollte
    es aber sein. Aus großer Kraft folgt große Verantwortung - einen guten Changelog zu
    besitzen heißt, ihn konsistent zu aktualisieren.

  %aside
    Es gibt sicher noch mehr. Hilf mir alle schlechten Angewohnheiten zu sammeln, indem du
    = link_to "ein Issue eröffnest", data.links.issue
    oder einen Pull-Request erstellst.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Häufig gestellte Fragen

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Gibt es ein standardisiertes Changelog-Format?

  %p
    Leider nicht. Es gibt zwar den #{link_to "GNU Changelog Styleguide", data.links.gnustyle} oder den
    #{link_to "zwei Absätze langen GNU NEWS-Datei \"Leitfaden\"", data.links.gnunews}. Beide sind aber
    unzureichend.

  %p
    Dieses Projekt versucht
    = link_to "eine bessere Changelog Konvention zu sein.", data.links.changelog
    Dazu beobachten wir bewährte Praktiken aus der Open Source Community
    und tragen sie zusammen.

  %p
    Gesunde Kritik, Diskussionen und Verbesserungsvorschläge
    = link_to "sind herzlich willkommen.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Wie sollte die Changelog-Datei benannt sein?

  %p
    Nenne sie <code>CHANGELOG.md</code>. Manche Projekte verwenden auch
    <code>HISTORY</code>, <code>NEWS</code> oder <code>RELEASES</code>.

  %p
    Man könnte zwar meinen, dass der Name der Changelog-Datei keine große
    Bedeutung hat, aber warum sollte man es den Endnutzern nicht einfach machen,
    die Änderungen des Projekts zu finden, indem man einen einheitlichen Namen
    verwendet?


  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Was ist mit GitHub Releases?

  %p
    Prinzipiell sind #{link_to "GitHub Releases", data.links.github_releases} eine gute Sache.
    Sie können dazu benutzt werden, um einfache Git Tags (zum Beispiel einen
    Tag namens <code>v1.0.0</code>) mit vielen Hinweisen zum Release
    auszustatten, indem man sie manuell bearbeitet, oder die Änderungen in eine
    Git Tag Nachricht schreibt, wo sie durch GitHub automatisch in die
    Release-Notizen gesetzt werden.

  %p
    Leider lassen sich GitHub Releases aber nicht so einfach exportieren
    und sind nur über GitHub selber zu lesen. Man kann sie auch so gestalten,
    dass sie dem Keep a Changelog Format sehr ähnlich sehen, aber dazu betreibt
    man in der Regel einen größeren Aufwand.

  %p
    Die derzeitige Version der GitHub Releases sind außerdem nicht leicht
    durch die Endnutzer zu finden, ganz im Gegenteil zu den typischerweise
    großgeschriebenen Dateien (<code>README</code>, <code>CONTRIBUTING</code>, etc.).
    Ein weiterer kleiner Nachteil ist, dass das Web Interface zurzeit keinen Link
    anbietet, um die Commits zwischen einzelnen Releases miteinander zu vergleichen.


  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Können Changelogs automatisiert ausgelesen werden?

  %p
    Es ist schwierig, weil Menschen meist unterschiedliche Formate und
    Dateinamen verwenden.

  %p
    #{link_to "Vandamme", data.links.vandamme} ist ein Ruby gem erstellt vom
    Gemnasium Team, das viele (aber nicht alle)
    Changelogs von Open-Source-Projekten einlesen kann.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Wie sieht es mit zurückgezogenen Releases aus?

  %p
    Sogenannte "Yanked Releases" sind Versionen, welche wegen schwerwiegenden
    Bugs oder Sicherheitsproblemen zurückgezogen werden mussten. Häufig kommen
    diese im Changelog gar nicht vor, aber das sollten sie. So solltest Du diese
    Versionen darstellen:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Die Kennzeichnung <code>[YANKED]</code> ist nicht ohne Grund großgeschrieben. Es ist wichtig,
    dass sie von Menschen bemerkt wird. Weil sie von eckigen Klammern umgeben ist,
    kann man sie außerdem einfacher automatisiert einlesen.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Sollte ich ein Changelog je umschreiben?

  %p
    Klar. Es gibt immer gute Gründe, ein Changelog zu verbessern. Ich öffne
    regelmässig Pull Requests, um Open-Source-Projekten mit schlecht gewarteten
    Changelogs fehlende Releases hinzuzufügen.

  %p
    Es ist auch möglich, dass Du eine wichtige Änderung vergessen hast, in einer
    Version zu erwähnen. Natürlich ist es in diesem Fall wichtig, das Changelog
    zu aktualisieren.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Wie kann ich mithelfen?

  %p
    Dieses Dokument ist nicht die <strong>Wahrheit</strong>. Es ist meine wohl überlegte Meinung,
    zusammen mit von mir zusammengetragenen Informationen und Beispielen.

  %p
    So möchte ich erreichen, dass die Community einen Konsens findet. Ich glaube, dass
    die Disskussion genauso wichtig ist wie das Endergebnis.

  %p
    Also bitte <strong>#{link_to "bring dich ein", data.links.repo}</strong>.

.press
  %h3 Gespräche
  %p
    Ich habe im #{link_to "The Changelog podcast", data.links.thechangelog} darüber gesprochen,
    warum sich Entwickler und Mitwirkende eines Projekts um ein Changelog kümmern sollten,
    sowie meine Motivationen hinter diesem Projekt erklärt.
```

## File: source/en/0.3.0/index.html.haml
```haml
---
title: Keep a Changelog
description: Don’t let your friends dump git logs into changelogs.
language: en
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### What’s a change log?
  A change log is a file which contains a curated, chronologically ordered
  list of notable changes for each version of a project.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### What’s the point of a change log?
  To make it easier for users and contributors to see precisely what
  notable changes have been made between each release (or version) of the project.

  ### Why should I care?
  Because software tools are for people. If you don’t care, why are
  you contributing to open source? Surely, there must be a kernel (ha!)
  of care somewhere in that lovely little brain of yours.

  I [talked with Adam Stacoviak and Jerod Santo on The Changelog][thechangelog]
  (fitting, right?) podcast about why maintainers and
  contributors should care, and the motivations behind this project.
  If you can spare the time (1:06), it’s a good listen.

  ### What makes a good change log?
  I’m glad you asked.

  A good change log sticks to these principles:

  - It’s made for humans, not machines, so legibility is crucial.
  - Easy to link to any section (hence Markdown over plain text).
  - One sub-section per version.
  - List releases in reverse-chronological order (newest on top).
  - Write all dates in `YYYY-MM-DD` format. (Example: `2012-06-02` for `June 2nd, 2012`.) It’s international, [sensible](https://xkcd.com/1179/), and language-independent.
  - Explicitly mention whether the project follows [Semantic Versioning][semver].
  - Each version should:
    - List its release date in the above format.
    - Group changes to describe their impact on the project, as follows:
      - `Added` for new features.
      - `Changed` for changes in existing functionality.
      - `Deprecated` for once-stable features removed in upcoming releases.
      - `Removed` for deprecated features removed in this release.
      - `Fixed` for any bug fixes.
      - `Security` to invite users to upgrade in case of vulnerabilities.

  ### How can I minimize the effort required?
  Always have an `"Unreleased"` section at the top for keeping track of any
  changes.

  This serves two purposes:

  - People can see what changes they might expect in upcoming releases
  - At release time, you just have to change `"Unreleased"` to the version number
    and add a new `"Unreleased"` header at the top.

  ### What makes unicorns cry?
  Alright…let’s get into it.

  - **Dumping a diff of commit logs.** Just don’t do that, you’re helping nobody.
  - **Not emphasizing deprecations.** When people upgrade from one version to
    another, it should be painfully clear when something will break.
  - **Dates in region-specific formats.** In the U.S., people put the month first
    ("06-02-2012" for June 2nd, 2012, which makes *no* sense), while many people
    in the rest of the world write a robotic-looking "2 June 2012", yet pronounce
    it differently. "2012-06-02" works logically from largest to smallest, doesn't
    overlap in ambiguous ways with other date formats, and is an
    [ISO standard](http://www.iso.org/iso/home/standards/iso8601.htm). Thus, it
    is the recommended date format for change logs.

  There’s more. Help me collect those unicorn tears by
  [opening an issue][issues]
  or a pull request.

  ### Is there a standard change log format?
  Sadly, no. Calm down. I know you're furiously rushing to find that link
  to the GNU change log style guide, or the two-paragraph GNU NEWS file
  "guideline". The GNU style guide is a nice start but it is sadly naive.
  There's nothing wrong with being naive but when people need
  guidance it's rarely very helpful. Especially when there are many
  situations and edge cases to deal with.

  This project [contains what I hope will become a better CHANGELOG file convention][CHANGELOG].
  I don't think the status quo is good enough, and I think that as a community we
  can come up with better conventions if we try to extract good practices from
  real software projects. Please take a look around and remember that
  [discussions and suggestions for improvements are welcome][issues]!

  ### What should the change log file be named?
  Well, if you can’t tell from the example above, `CHANGELOG.md` is the
  best convention so far.

  Some projects also use `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`, etc.

  It’s a mess. All these names only makes it harder for people to find it.

  ### Why can’t people just use a `git log` diff?
  Because log diffs are full of noise — by nature. They could not make a suitable
  change log even in a hypothetical project run by perfect humans who never make
  typos, never forget to commit new files, never miss any part of a refactoring.
  The purpose of a commit is to document one atomic step in the process by which
  the code evolves from one state to another. The purpose of a change log is to
  document the noteworthy differences between these states.

  As is the difference between good comments and the code itself,
  so is the difference between a change log and the commit log:
  one describes the *why*, the other the how.

  ### Can change logs be automatically parsed?
  It’s difficult, because people follow wildly different formats and file names.

  [Vandamme][vandamme] is a Ruby gem
  created by the [Gemnasium][gemnasium] team and which parses
  many (but not all) open source project change logs.

  ### Why do you alternate between spelling it "CHANGELOG" and "change log"?
  "CHANGELOG" is the name of the file itself. It's a bit shouty but it's a
  historical convention followed by many open source projects. Other
  examples of similar files include [`README`][README], [`LICENSE`][LICENSE],
  and [`CONTRIBUTING`][CONTRIBUTING].

  The uppercase naming (which in old operating systems made these files stick
  to the top) is used to draw attention to them. Since they're important
  metadata about the project, they could be useful to anyone intending to use
  or contribute to it, much like [open source project badges][shields].

  When I refer to a "change log", I'm talking about the function of this
  file: to log changes.

  ### What about yanked releases?
  Yanked releases are versions that had to be pulled because of a serious
  bug or security issue. Often these versions don't even appear in change
  logs. They should. This is how you should display them:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  The `[YANKED]` tag is loud for a reason. It's important for people to
  notice it. Since it's surrounded by brackets it's also easier to parse
  programmatically.

  ### Should you ever rewrite a change log?
  Sure. There are always good reasons to improve a change log. I regularly open
  pull requests to add missing releases to open source projects with unmaintained
  change logs.

  It's also possible you may discover that you forgot to address a breaking change
  in the notes for a version. It's obviously important for you to update your
  change log in this case.

  ### How can I contribute?
  This document is not the **truth**; it’s my carefully considered
  opinion, along with information and examples I gathered.
  Although I provide an actual [CHANGELOG][] on [the GitHub repo][gh],
  I have purposefully not created a proper *release* or clear list of rules
  to follow (as [SemVer.org][semver] does, for instance).

  This is because I want our community to reach a consensus. I believe the
  discussion is as important as the end result.

  So please [**pitch in**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/en/1.0.0/index.html.haml
```haml
---
title: Keep a Changelog
description: Don’t let your friends dump git logs into changelogs.
language: en
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    What is a changelog?

  %p
    A changelog is a file which contains a curated, chronologically
    ordered list of notable changes for each version of a project.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Why keep a changelog?

  %p
    To make it easier for users and contributors to see precisely what
    notable changes have been made between each release (or version) of
    the project.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Who needs a changelog?

  %p
    People do. Whether consumers or developers, the end users of
    software are human beings who care about what's in the software. When
    the software changes, people want to know why and how.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    How do I make a good changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Guiding Principles

  %ul
    %li
      Changelogs are <em>for humans</em>, not machines.
    %li
      There should be an entry for every single version.
    %li
      The same types of changes should be grouped.
    %li
      Versions and sections should be linkable.
    %li
      The latest version comes first.
    %li
      The release date of each version is displayed.
    %li
      Mention whether you follow #{link_to "Semantic Versioning", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Types of changes

  %ul
    %li
      %code Added
      for new features.
    %li
      %code Changed
      for changes in existing functionality.
    %li
      %code Deprecated
      for soon-to-be removed features.
    %li
      %code Removed
      for now removed features.
    %li
      %code Fixed
      for any bug fixes.
    %li
      %code Security
      in case of vulnerabilities.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    How can I reduce the effort required to maintain a changelog?

  %p
    Keep an <code>Unreleased</code> section at the top to track upcoming
    changes.

  %p This serves two purposes:

  %ul
    %li
      People can see what changes they might expect in upcoming releases
    %li
      At release time, you can move the <code>Unreleased</code> section
      changes into a new release version section.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Can changelogs be bad?

  %p Yes. Here are a few ways they can be less than useful.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    Using commit log diffs as changelogs is a bad idea: they're full of
    noise. Things like merge commits, commits with obscure titles,
    documentation changes, etc.

  %p
    The purpose of a commit is to document a step in the evolution of
    the source code. Some projects clean up commits, some don't.

  %p
    The purpose of a changelog entry is to document the noteworthy
    difference, often across multiple commits, to communicate them
    clearly to end users.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignoring Deprecations

  %p
    When people upgrade from one version to another, it should be
    painfully clear when something will break. It should be possible to
    upgrade to a version that lists deprecations, remove what's
    deprecated, then upgrade to the version where the deprecations
    become removals.

  %p
    If you do nothing else, list deprecations, removals, and any
    breaking changes in your changelog.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Confusing Dates

  %p
    Regional date formats vary throughout the world and it's often
    difficult to find a human-friendly date format that feels intuitive
    to everyone. The advantage of dates formatted like
    <code>2017-07-17</code> is that they follow the order of largest to
    smallest units: year, month, and day. This format also doesn't
    overlap in ambiguous ways with other date formats, unlike some
    regional formats that switch the position of month and day numbers.
    These reasons, and the fact this date format is an
    #{link_to "ISO standard", data.links.isodate}, are why it is the recommended date
    format for changelog entries.

  %aside
    There’s more. Help me collect these antipatterns by
    = link_to "opening an issue", data.links.issue
    or a pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Frequently Asked Questions

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Is there a standard changelog format?

  %p
    Not really. There's the #{link_to "GNU changelog style guide", data.links.gnustyle},
    or the #{link_to "two-paragraph-long GNU NEWS file", data.links.gnunews}
    "guideline". Both are inadequate or insufficient.

  %p
    This project aims to be
    = link_to "a better changelog convention.", data.links.changelog
    It comes from observing good practices in the open source
    community and gathering them.

  %p
    Healthy criticism, discussion and suggestions for improvements
    = link_to "are welcome.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    What should the changelog file be named?

  %p
    Call it <code>CHANGELOG.md</code>. Some projects use
    <code>HISTORY</code>, <code>NEWS</code> or <code>RELEASES</code>.

  %p
    While it's easy to think that the name of your changelog file
    doesn't matter that much, why make it harder for your end users to
    consistently find notable changes?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    What about GitHub Releases?

  %p
    It's a great initiative. #{link_to "Releases", data.links.github_releases} can be used to
    turn simple git tags (for example a tag named <code>v1.0.0</code>)
    into rich release notes by manually adding release notes or it can
    pull annotated git tag messages and turn them into notes.

  %p
    GitHub Releases create a non-portable changelog that can only be
    displayed to users within the context of GitHub. It's possible to
    make them look very much like the Keep a Changelog format, but it
    tends to be a bit more involved.

  %p
    The current version of GitHub releases is also arguably not very
    discoverable by end-users, unlike the typical uppercase files
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.). Another
    minor issue is that the interface doesn't currently offer links to
    commit logs between each release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Can changelogs be automatically parsed?

  %p
    It’s difficult, because people follow wildly different formats and
    file names.

  %p
    #{link_to "Vandamme", data.links.vandamme} is a Ruby gem created by the
    Gemnasium team and which parses many (but
    not all) open source project changelogs.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    What about yanked releases?

  %p
    Yanked releases are versions that had to be pulled because of a
    serious bug or security issue. Often these versions don't even
    appear in change logs. They should. This is how you should display
    them:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    The <code>[YANKED]</code> tag is loud for a reason. It's important
    for people to notice it. Since it's surrounded by brackets it's also
    easier to parse programmatically.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Should you ever rewrite a changelog?

  %p
    Sure. There are always good reasons to improve a changelog. I
    regularly open pull requests to add missing releases to open source
    projects with unmaintained changelogs.

  %p
    It's also possible you may discover that you forgot to address a
    breaking change in the notes for a version. It's obviously important
    for you to update your changelog in this case.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    How can I contribute?

  %p
    This document is not the <strong>truth</strong>; it’s my carefully
    considered opinion, along with information and examples I gathered.

  %p
    This is because I want our community to reach a consensus. I believe
    the discussion is as important as the end result.

  %p
    So please <strong>#{link_to "pitch in", data.links.repo}</strong>.

.press
  %h3 Conversations
  %p
    I went on #{link_to "The Changelog podcast", data.links.thechangelog}
    to talk about why maintainers and contributors should care about changelogs,
    and also about the motivations behind this project.
```

## File: source/en/1.1.0/index.html.haml
```haml
---
title: Keep a Changelog
description: Don’t let your friends dump git logs into changelogs.
language: en
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    What is a changelog?

  %p
    A changelog is a file which contains a curated, chronologically
    ordered list of notable changes for each version of a project.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Why keep a changelog?

  %p
    To make it easier for users and contributors to see precisely what
    notable changes have been made between each release (or version) of
    the project.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Who needs a changelog?

  %p
    People do. Whether consumers or developers, the end users of
    software are human beings who care about what's in the software. When
    the software changes, people want to know why and how.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    How do I make a good changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Guiding Principles

  %ul
    %li
      Changelogs are <em>for humans</em>, not machines.
    %li
      There should be an entry for every single version.
    %li
      The same types of changes should be grouped.
    %li
      Versions and sections should be linkable.
    %li
      The latest version comes first.
    %li
      The release date of each version is displayed.
    %li
      Mention whether you follow #{link_to "Semantic Versioning", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Types of changes

  %ul
    %li
      %code Added
      for new features.
    %li
      %code Changed
      for changes in existing functionality.
    %li
      %code Deprecated
      for soon-to-be removed features.
    %li
      %code Removed
      for now removed features.
    %li
      %code Fixed
      for any bug fixes.
    %li
      %code Security
      in case of vulnerabilities.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    How can I reduce the effort required to maintain a changelog?

  %p
    Keep an <code>Unreleased</code> section at the top to track upcoming
    changes.

  %p This serves two purposes:

  %ul
    %li
      People can see what changes they might expect in upcoming releases
    %li
      At release time, you can move the <code>Unreleased</code> section
      changes into a new release version section.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Can changelogs be bad?

  %p Yes. Here are a few ways they can be less than useful.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    Using commit log diffs as changelogs is a bad idea: they're full of
    noise. Things like merge commits, commits with obscure titles,
    documentation changes, etc.

  %p
    The purpose of a commit is to document a step in the evolution of
    the source code. Some projects clean up commits, some don't.

  %p
    The purpose of a changelog entry is to document the noteworthy
    difference, often across multiple commits, to communicate them
    clearly to end users.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignoring Deprecations

  %p
    When people upgrade from one version to another, it should be
    painfully clear when something will break. It should be possible to
    upgrade to a version that lists deprecations, remove what's
    deprecated, then upgrade to the version where the deprecations
    become removals.

  %p
    If you do nothing else, list deprecations, removals, and any
    breaking changes in your changelog.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Confusing Dates

  %p
    Regional date formats vary throughout the world and it's often
    difficult to find a human-friendly date format that feels intuitive
    to everyone. The advantage of dates formatted like
    <code>2017-07-17</code> is that they follow the order of largest to
    smallest units: year, month, and day. This format also doesn't
    overlap in ambiguous ways with other date formats, unlike some
    regional formats that switch the position of month and day numbers.
    These reasons, and the fact this date format is an
    #{link_to "ISO standard", data.links.isodate}, are why it is the recommended date
    format for changelog entries.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Inconsistent Changes

  %p
    A changelog which only mentions some of the changes can be as dangerous
    as not having a changelog. While many of the changes may not be
    relevant - for instance, removing a single whitespace may not need
    to be recorded in all instances - any important changes should be
    mentioned in the changelog. By inconsistently applying changes,
    your users may mistakenly think that the changelog is the single source
    of truth. It ought to be. With great power comes great responsibility -
    having a good changelog means having a consistently updated changelog.

  %aside
    There’s more. Help me collect these antipatterns by
    = link_to "opening an issue", data.links.issue
    or a pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Frequently Asked Questions

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Is there a standard changelog format?

  %p
    Not really. There's the #{link_to "GNU changelog style guide", data.links.gnustyle},
    or the #{link_to "two-paragraph-long GNU NEWS file", data.links.gnunews}
    "guideline". Both are inadequate or insufficient.

  %p
    This project aims to be
    = link_to "a better changelog convention.", data.links.changelog
    It comes from observing good practices in the open source
    community and gathering them.

  %p
    Healthy criticism, discussion and suggestions for improvements
    = link_to "are welcome.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    What should the changelog file be named?

  %p
    Call it <code>CHANGELOG.md</code>. Some projects use
    <code>HISTORY</code>, <code>NEWS</code> or <code>RELEASES</code>.

  %p
    While it's easy to think that the name of your changelog file
    doesn't matter that much, why make it harder for your end users to
    consistently find notable changes?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    What about GitHub Releases?

  %p
    It's a great initiative. #{link_to "Releases", data.links.github_releases} can be used to
    turn simple git tags (for example a tag named <code>v1.0.0</code>)
    into rich release notes by manually adding release notes or it can
    pull annotated git tag messages and turn them into notes.

  %p
    GitHub Releases create a non-portable changelog that can only be
    displayed to users within the context of GitHub. It's possible to
    make them look very much like the Keep a Changelog format, but it
    tends to be a bit more involved.

  %p
    The current version of GitHub releases is also arguably not very
    discoverable by end-users, unlike the typical uppercase files
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.). Another
    minor issue is that the interface doesn't currently offer links to
    commit logs between each release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Can changelogs be automatically parsed?

  %p
    It’s difficult, because people follow wildly different formats and
    file names.

  %p
    #{link_to "Vandamme", data.links.vandamme} is a Ruby gem created by the
    Gemnasium team and which parses many (but
    not all) open source project changelogs.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    What about yanked releases?

  %p
    Yanked releases are versions that had to be pulled because of a
    serious bug or security issue. Often these versions don't even
    appear in change logs. They should. This is how you should display
    them:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    The <code>[YANKED]</code> tag is loud for a reason. It's important
    for people to notice it. Since it's surrounded by brackets it's also
    easier to parse programmatically.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Should you ever rewrite a changelog?

  %p
    Sure. There are always good reasons to improve a changelog. I
    regularly open pull requests to add missing releases to open source
    projects with unmaintained changelogs.

  %p
    It's also possible you may discover that you forgot to address a
    breaking change in the notes for a version. It's obviously important
    for you to update your changelog in this case.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    How can I contribute?

  %p
    This document is not the <strong>truth</strong>; it’s my carefully
    considered opinion, along with information and examples I gathered.

  %p
    This is because I want our community to reach a consensus. I believe
    the discussion is as important as the end result.

  %p
    So please <strong>#{link_to "pitch in", data.links.repo}</strong>.

.press
  %h3 Conversations
  %p
    I went on #{link_to "The Changelog podcast", data.links.thechangelog}
    to talk about why maintainers and contributors should care about changelogs,
    and also about the motivations behind this project.
```

## File: source/es-ES/0.3.0/index.html.haml
```haml
---
title: Mantenga un Changelog
description: No dejes que tus amigos copien y peguen git logs en los CHANGELOGs
language: es-ES
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Qué es un registro de cambios (change log)?

  Un registro de cambios o “change log” de ahora en adelante, es un archivo que contiene una lista en orden cronológico sobre los cambios que vamos haciendo en cada reléase (o versión) de nuestro proyecto.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Cuál es el propósito del change log?

  Para que les sea más fácil a los usuarios y contribuyentes, ver exactamente los cambios notables que se han hecho entre cada versión (o versiones) del proyecto.

  ### Por qué me debería importar?

  Debido a que las herramientas de software son para la gente. Si no te importa, ¿por qué contribuyes al código abierto? Sin duda, tiene que haber un "kernel" (ha!) de importancia en ese pequeño y encantador cerebro tuyo.

  [En el podcast "The Changelog" hablé con Adam Stacoviak y Jerod Santo][thechangelog]
  (muy apropiado, ¿no?) acerca de por qué nos debería importar y sobre las motivaciones que es están detrás del proyecto. Si tienes tiempo (1:06), escúchalo, vale la pena

  ### Cómo podemos hacer un buen change log?

  Me alegro de que te lo hayas preguntado.

  Un buen change log se guía por los siguientes principios:

  - Está hecho para los seres humanos, no máquinas, por lo que la legibilidad es fundamental.
  - Fácil de conectar a cualquier sección (de ahí a que se use Markdown sobre texto plano).
  - Una sub-sección por versión.
  - Lista los releases en un orden inversamente-cronológico (el más reciente en la parte superior).
  - Escribe todas las fechas en formato `AAAA-MM-DD`. (Ejemplo: `2012-06-02` en vez de `2 JUN 2012`.) Es internacional, [sensible](https://xkcd.com/1179/), e independientemente del lenguaje que usemos.
  - Se menciona explícitamente si el proyecto sigue la convención del [Versionamiento Semántico][semver].
  - Cada versión debería:
    - Indicar su fecha de lanzamiento en el formato anterior.
    - Agrupar los cambios para describir su impacto en el proyecto, de la siguiente manera:
      - `Added` para funcionalidades nuevas.
      - `Changed` para los cambios en las funcionalidades existentes.
      - `Deprecated` para indicar que una característica está obsoleta y que se eliminará en las próximas versiones.
      - `Removed` para las características en desuso que se eliminaron en esta versión.
      - `Fixed` para correcciones y bugs.
      - `Security` para invitar a los usuarios a actualizar, en el caso de que haya vulnerabilidades.

  ### Cómo puedo minimizar el esfuerzo requerido?

  Siempre mantén una sección con el nombre `"Unreleased"` para hacer un seguimiento sobre los cambios

  Esto nos puede servir para dos cosas:

  - La gente puede ver qué cambios podrían esperar en los próximos releases
  - Una vez queramos hacer una release, sólo hay que cambiar `Unreleased` por el número de versión y añadir una nueva cabecera `Unreleased` en la parte superior.

  ### Qué es lo que hace llorar a los unicornios?

  Muy bien... vamos allá.

  - **Hacer un copia y pega de un diff o de los logs de los commits.** Simplemente no lo hagas, no estas ayudando a nadie.
  - **No indicar las características obsoletas.** Cuando la gente actualiza de una versión a otra,debe tener claro cuando algo se romperá.
  - **Fechas en formatos específicos de una región.** En los EE.UU., la gente pone primero el mes ("06/02/2012" para el 2 de junio del 2012, lo que hace no tiene sentido), mientras que muchas personas en el resto del mundo escriben de una manera muy robótica "2 JUN 2012". "2012-06-02" me parece más lógico, y también cabe comentar que es un [ISO standard](http://www.iso.org/iso/home/standards/iso8601.htm). Por lo tanto, es el formato de la fecha recomendada para los change logs

  Pero espera! hay más ayúdame a coleccionar esas lágrimas de unicornio [abriendo una incidencia][issues] o haciendo un pull request.

  ### Hay algún formato estándar de formato para los change log?

  Tristemente, no. Pero calma. Sé que estás corriendo furiosamente intentando encontrar ese link al libro de estilo de registro de cambios de GNU, or the two-paragraph GNU NEWS file
  "guideline". La guía de estilo GNU es un buen comienzo, pero es tristemente cándida. No hay nada malo en ser cándida, pero cuando la gente necesita orientación es rara la vez, que resulta ser muy útil. Sobre todo, cuando hay muchas situaciones y casos muy específicos.

  Este proyecto [contiene lo que espero se convierta en un mejor patrón de CHANGELOGs][CHANGELOG].

  No creo que la situación actual sea lo suficientemente buena, i creo que como comunidad que somos podemos llegar a mejores convenciones si tratamos de extraer buenas prácticas de proyectos de software reales. Por favor echa un pequeño vistazo y recuerda que las [sugerencias y discusiones para mejorar son bienvenidas][issues]!

  ### Cómo se debería llamar el change log?

  Bueno, si te fijas en en ejemplo anterior, `CHANGELOG.md` es la convención más usada

  Otros proyectos también usan los siguientes nombres `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`, etc.

  Es un desastre. Todos estos nombres sólo hacen más difícil la búsqueda del fichero.

  ### Por qué la gente no usa simplemente un `git log`?

  Debido a que están llenos de ruido - por naturaleza. No se podría hacer un change log adecuado ni siquiera en un proyecto hipotético dirigido por seres humanos perfectos que nunca se equivocan y que nunca se olvidan meter ningún archivo en un commit... etc. El propósito de un commit es el de documentar un cambio atómico en el cual el software evoluciona desde un estado hacia otro. El propósito del change log es el de documentar las diferencias notables entre estos estados.

  ### Se pueden parsear automáticamente los change logs?

  Es difícil, ya que la gente sigue formatos y nombres de archivo muy distintos.

  [Vandamme][vandamme] es un Ruby gem creado por el equipo [Gemnasium][gemnasium], que lo que hace es parsear algunos (no todos) los change logs de varios proyectos open source.

  ### Por que estás continuamente alternando los nombres de "CHANGELOG" a "change log"?

  "CHANGELOG" es el nombre del archivo en sí. Es un poco intrusivo pero es una convención histórica seguida por muchos proyectos de código abierto. Otro ejemplo de este tipo de nombres en archivos son [`README`][README], [`LICENSE`][LICENSE],
  y [`CONTRIBUTING`][CONTRIBUTING].

  Los nombres en mayúsculas (que en algunos sistemas operativos antiguos hacían que estos ficheros aparecieran los primeros) se utilizan para llamar la atención sobre ellos. Dado que son importantes metadatos sobre el proyecto, que podría ser útil a cualquier persona con la intención de utilizar o contribuir al mismo.

  Cuando me refiero a "change log", estoy hablando de la función del fichero: registrar los cambios.

  ### Qué son las yanked releases?

  Las yanked releases son versiones que tuvieron que ser retiradas a causa de un grave error o problema de seguridad. A menudo, estas versiones ni siquiera aparecen en los change logs, y tendrían que aparecer. Así es como se muestran:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  La sección `[YANKED]` va entre corchetes por una razón, es importante que destaque, y el echo de estar rodeado por corchetes lo hace más fácil de localizar programáticamente.

  ### Deberías volver a escribir un change log?

  Por supuesto. Siempre hay buenas razones para mejorar el change log. A veces abro "pull requests" para añadir registros faltantes en el change log de proyectos open source.

  ### Como puedo contribuir?

  Este documento no es la **verdad absoluta**; es mi cuidadosa opinión, junto con información y ejemplos que recogí.

  Esto es porque quiero que la comunidad llegue a un conceso. Creo que la discusión es tan importante como el resultado final.

  [**Contribuye**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/es-ES/1.0.0/index.html.haml
```haml
---
title: Mantenga un changelog
description: No dejes que tus amigos usen el registro de git en los changelogs.
language: es-ES
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2
      No dejes que tus amigos usen el registro de git en los changelogs.

  = link_to data.links.changelog do
    Versión
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    ¿Qué es un registro de cambios (changelog)?

  %p
    Un registro de cambios, «changelog» de ahora en adelante, es un
    archivo que contiene una lista cronológicamente ordenada de los
    cambios más destacables para cada versión de un proyecto.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    ¿Por qué mantener un changelog?

  %p
    Para facilitar a los usuarios y colaboradores ver exactamente qué
    cambios reseñables se han realizado entre cada versión del
    proyecto.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    ¿Quién necesita un changelog?

  %p
    Las personas. Ya sean consumidores o desarrolladores, los usuarios
    finales del software son seres humanos a los que le importa lo que
    hay en el software. Cuando el software cambia, la gente quiere saber
    el porqué y el cómo.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    ¿Cómo hago un buen changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Directrices

  %ul
    %li
      Están hechos <em>para los seres humanos</em>, no para las máquinas.
    %li
      Debe haber una entrada para cada versión.
    %li
      Los mismos tipos de cambios deben ser agrupados.
    %li
      Versiones y secciones deben ser enlazables.
    %li
      La última versión va primero.
    %li
      Debe mostrar la fecha de publicación de cada versión.
    %li
      Indicar si el proyecto sigue el #{link_to "Versionamiento Semántico", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipos de cambios

  %ul
    %li
      %code Added
      para funcionalidades nuevas.
    %li
      %code Changed
      para los cambios en las funcionalidades existentes.
    %li
      %code Deprecated
      para indicar que una característica o funcionalidad está obsoleta
      y que se eliminará en las próximas versiones.
    %li
      %code Removed
      para las características en desuso que se eliminaron en esta
      versión.
    %li
      %code Fixed
      para corrección de errores.
    %li
      %code Security
      en caso de vulnerabilidades.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    ¿Cómo puedo minimizar el esfuerzo requerido para mantener el
    changelog?

  %p
    Mantén una sección con el nombre <code>Unreleased</code> para hacer
    un seguimiento sobre los próximos cambios.

  %p Esto nos puede servir para dos cosas:

  %ul
    %li
      La gente puede ver qué cambios podrían esperar en los próximos
      lanzamientos.
    %li
      Al lanzar una nueva versión, tan sólo habría que mover el
      contenido de <code>Unreleased</code> a una sección para la nueva
      versión.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    ¿Pueden los changelogs ser malos?

  %p
    Sí. A continuación algunas formas en las que pueden ser muy poco
    útiles.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Usar un diff de los logs de los commits

  %p
    Usar un diff de los logs de los commits es una mala idea: están
    llenos de ruido. Cosas como hacer <em>merge</em> de los commits,
    commits con títulos poco claros, cambios de documentación, etc.

  %p
    El propósito de un commit es documentar un paso en la evolución del
    código fuente. Algunos proyectos limpian los commits, otros no.

  %p
    El propósito de una entrada en el changelog es documentar cambios
    notables, usualmente entre múltiples commits, para comunicarlos
    claramente a los usuarios finales.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorar <em>funcionalidades sin soporte</em>

  %p
    Cuando la gente actualiza de una version a otra, debería estar
    bastante claro cuándo algo se va a romper. Debería ser posible
    actualizar a una versión que detalle funcionalidades sin soporte,
    eliminar lo que está obsoleto y actualizar a la versión donde esas
    funcionalidades han sido eliminadas.
  %p
    Si no haces nada más, enumera lo que queda obsoleto, lo eliminado y
    cualquier otro cambio sin compatibilidad hacia atrás en tu
    changelog.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Fechas confusas

  %p
    Los formatos de fecha regionales varían en todo el mundo y con
    frecuencia es difícil encontrar un formato intuitivo para todos. La
    ventaja de las fechas con el formato <code>2017-07-17</code> es que
    siguen un orden de unidades de mayor a menor: año, mes y día. Este
    formato tampoco coincide de forma ambigua con otros formatos de
    fecha, a diferencia de algunos que intercambian la posición del mes
    y el día. Todo esto, junto al hecho de ser un
    #{link_to "estándar ISO", data.links.isodate}, son los motivos por los que es el
    formato de fecha recomendado para las entradas del changelog.

  %aside
    Hay más. Ayúdame a recoger estos anti-patrones
    = link_to "abriendo un issue", data.links.issue
    o una <em>pull request</em>.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Preguntas frecuentes

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    ¿Hay un formato estándar para el changelog?

  %p
    No. Hay una guía de estilo del GNU o los dos parrafos del archivo
    <code>guideline</code> del <em>GNU NEWS</em>. Ambos son inadecuados
    o insuficientes.

  %p
    Este proyecto apunta a ser
    = link_to "una mejor convención de changelog.", data.links.changelog
    Esto se da observando las buenas prácticas en la comunidad open
    source y recopilando las mismas.

  %p
    Críticas saludables, discusión y sugerencias para mejoras
    = link_to "son bienvenidas.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    ¿Cómo debería llamarse el archivo de changelog?

  %p
    Llámalo <code>CHANGELOG.md</code>. Algunos proyectos utilizan
    <code>HISTORY</code>, <code>NEWS</code> o <code>RELEASES</code>.

  %p
    Si bien es fácil pensar que el nombre de tu archivo de changelog no
    importa tanto, ¿Por qué hacer difícil para los usuarios finales
    conseguir de manera consistente los cambios notables?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    ¿Y qué hay de los releases de Github?

  %p
    Es una gran iniciativa. #{link_to "Los releases de Github", data.links.github_releases}
    pueden ser utilizados para convertir simples etiquetas de git (por
    ejemplo una etiqueta llamada <code>v1.0.0</code>) en ricas notas de
    lanzamiento ya sea añadiendo estas manualmente o trayendo los
    mensajes anotados de las etiquetas de git y convertirlas en notas.

  %p
    Los releases de Github crean un changelog no portable que sólo
    pueden ser mostrados a usuarios dentro del contexto de Github. Es
    posible hacer que luzcan muy parecidas al formato de <em>Mantenga un
    changelog</em>, pero tiende a ser un poco más complicado.

  %p
    La versión actual de los releases de Github es también
    discutiblemente no muy detectable por los usuarios finales,
    diferente a los típicos archivos en mayúsculas (<code>README</code>,
    <code>CONTRIBUTING</code>, etc.). Otro problema menor es que la
    interfaz actualmente no ofrece enlaces a los registros de los
    commits entre cada lanzamiento.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    ¿Se pueden analizar gramaticalmente los changelogs?

  %p
    Es difícil, porque las personas usan formatos y nombres de archivos
    muy distintos.

  %p
    #{link_to "Vandamme", data.links.vandamme} es una gema de Ruby creada por el
    equipo de Gemnasium que analiza
    gramaticalmente muchos (pero no todos) los changelogs de proyectos
    open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    ¿Qué hay sobre las versiones retiradas?

  %p
    «Yanked releases» son versiones que tuvieron que ser retiradas por
    un error grave o problema de seguridad. Con frecuencia estas
    versiones ni siquiera aparecen en los changelogs. Deberían. Así es
    como deberían mostrarse:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    La etiqueta <code>[YANKED]</code> está destacada por una razón: es
    importante que destaque. El hecho de estar entre corchetes la hace
    también más fácil de localizar programáticamente.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    ¿Deberías volver a escribir un changelog?

  %p
    Por supuesto. Siempre hay buenas razones para mejorar el changelog.
    A veces abro «pull requests» para añadir registros faltantes en el
    changelog de proyectos open source.

  %p
    También es posible que puedas descubrir que olvidaste señalar un
    cambio sin compatibilidad hacia atrás en las notas para una versión.
    En este caso es importante para ti actualizar el changelog.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    ¿Cómo puedo contribuir?

  %p
    Este documento no es la <strong>verdad absoluta</strong>; es mi
    cuidadosa opinión, junto con información y ejemplos que recopilé.

  %p
    Esto es porque quiero que la comunidad llegue a un consenso. Creo
    que la discusión es tan importante como el resultado final.

  %p
    Así que por favor <strong>#{link_to "comienza a colaborar", data.links.repo}
    </strong>.

.press
  %h3 Conversaciones
  %p
    Fui a #{link_to "The Changelog podcast", data.links.thechangelog} para hablar
    acerca del porqué los mantenedores y colaboradores deberían
    preocuparse por los changelogs, y también acerca de las motivaciones
    detrás de este proyecto.
```

## File: source/es-ES/1.1.0/index.html.haml
```haml
---
title: Mantenga un changelog
description: No dejes que tus amigos usen el registro de git en los changelogs.
language: es-ES
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2
      No dejes que tus amigos usen el registro de git en los changelogs.

  = link_to data.links.changelog do
    Versión
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    ¿Qué es un registro de cambios (changelog)?

  %p
    Un registro de cambios, «changelog» de ahora en adelante, es un
    archivo que contiene una lista cronológicamente ordenada de los
    cambios más destacables para cada versión de un proyecto.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    ¿Por qué mantener un changelog?

  %p
    Para facilitar a los usuarios y colaboradores ver exactamente qué
    cambios reseñables se han realizado entre cada versión del
    proyecto.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    ¿Quién necesita un changelog?

  %p
    Las personas. Ya sean consumidores o desarrolladores, los usuarios
    finales del software son seres humanos a los que le importa lo que
    hay en el software. Cuando el software cambia, la gente quiere saber
    el porqué y el cómo.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    ¿Cómo hago un buen changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Directrices

  %ul
    %li
      Están hechos <em>para los seres humanos</em>, no para las máquinas.
    %li
      Debe haber una entrada para cada versión.
    %li
      Los mismos tipos de cambios deben ser agrupados.
    %li
      Versiones y secciones deben ser enlazables.
    %li
      La última versión va primero.
    %li
      Debe mostrar la fecha de publicación de cada versión.
    %li
      Indicar si el proyecto sigue el #{link_to "Versionamiento Semántico", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipos de cambios

  %ul
    %li
      %code Added
      para funcionalidades nuevas.
    %li
      %code Changed
      para los cambios en las funcionalidades existentes.
    %li
      %code Deprecated
      para indicar que una característica o funcionalidad está obsoleta
      y que se eliminará en las próximas versiones.
    %li
      %code Removed
      para las características en desuso que se eliminaron en esta
      versión.
    %li
      %code Fixed
      para corrección de errores.
    %li
      %code Security
      en caso de vulnerabilidades.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    ¿Cómo puedo minimizar el esfuerzo requerido para mantener el
    changelog?

  %p
    Mantén una sección con el nombre <code>Unreleased</code> para hacer
    un seguimiento sobre los próximos cambios.

  %p Esto nos puede servir para dos cosas:

  %ul
    %li
      La gente puede ver qué cambios podrían esperar en los próximos
      lanzamientos.
    %li
      Al lanzar una nueva versión, tan sólo habría que mover el
      contenido de <code>Unreleased</code> a una sección para la nueva
      versión.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    ¿Pueden los changelogs ser malos?

  %p
    Sí. A continuación algunas formas en las que pueden ser muy poco
    útiles.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Usar un diff de los logs de los commits

  %p
    Usar un diff de los logs de los commits es una mala idea: están
    llenos de ruido. Cosas como hacer <em>merge</em> de los commits,
    commits con títulos poco claros, cambios de documentación, etc.

  %p
    El propósito de un commit es documentar un paso en la evolución del
    código fuente. Algunos proyectos limpian los commits, otros no.

  %p
    El propósito de una entrada en el changelog es documentar cambios
    notables, usualmente entre múltiples commits, para comunicarlos
    claramente a los usuarios finales.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorar <em>funcionalidades sin soporte</em>

  %p
    Cuando la gente actualiza de una version a otra, debería estar
    bastante claro cuándo algo se va a romper. Debería ser posible
    actualizar a una versión que detalle funcionalidades sin soporte,
    eliminar lo que está obsoleto y actualizar a la versión donde esas
    funcionalidades han sido eliminadas.
  %p
    Si no haces nada más, enumera lo que queda obsoleto, lo eliminado y
    cualquier otro cambio sin compatibilidad hacia atrás en tu
    changelog.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Fechas confusas

  %p
    Los formatos de fecha regionales varían en todo el mundo y con
    frecuencia es difícil encontrar un formato intuitivo para todos. La
    ventaja de las fechas con el formato <code>2017-07-17</code> es que
    siguen un orden de unidades de mayor a menor: año, mes y día. Este
    formato tampoco coincide de forma ambigua con otros formatos de
    fecha, a diferencia de algunos que intercambian la posición del mes
    y el día. Todo esto, junto al hecho de ser un
    #{link_to "estándar ISO", data.links.isodate}, son los motivos por los que es el
    formato de fecha recomendado para las entradas del changelog.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Cambios inconsistentes

  %p
    Un changelog que solo menciona algunos de los cambios puede ser tan
    peligroso como no tener un changelog. Mientras muchos de los cambios
    puede no ser relavantes - por ejemplo, eliminar un simple espacio en
    blanco - cualquier cambio importante debe ser mencionado. Al aplicar
    cambios incoherentes, tus usuarios pueden pensar que el changelog es
    la única fuente de verdad. Y debería serlo. Un gran poder conlleva una
    gran responsabilidad - tener un buen changelog significa mantenerlo
    actualizado.

  %aside
    Hay más. Ayúdame a recoger estos anti-patrones
    = link_to "abriendo un issue", data.links.issue
    o una <em>pull request</em>.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Preguntas frecuentes

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    ¿Hay un formato estándar para el changelog?

  %p
    No. Hay una guía de estilo del GNU o los dos parrafos del archivo
    <code>guideline</code> del <em>GNU NEWS</em>. Ambos son inadecuados
    o insuficientes.

  %p
    Este proyecto apunta a ser
    = link_to "una mejor convención de changelog.", data.links.changelog
    Esto se da observando las buenas prácticas en la comunidad open
    source y recopilando las mismas.

  %p
    Críticas saludables, discusión y sugerencias para mejoras
    = link_to "son bienvenidas.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    ¿Cómo debería llamarse el archivo de changelog?

  %p
    Llámalo <code>CHANGELOG.md</code>. Algunos proyectos utilizan
    <code>HISTORY</code>, <code>NEWS</code> o <code>RELEASES</code>.

  %p
    Si bien es fácil pensar que el nombre de tu archivo de changelog no
    importa tanto, ¿Por qué hacer difícil para los usuarios finales
    conseguir de manera consistente los cambios notables?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    ¿Y qué hay de los releases de Github?

  %p
    Es una gran iniciativa. #{link_to "Los releases de Github", data.links.github_releases}
    pueden ser utilizados para convertir simples etiquetas de git (por
    ejemplo una etiqueta llamada <code>v1.0.0</code>) en ricas notas de
    lanzamiento ya sea añadiendo estas manualmente o trayendo los
    mensajes anotados de las etiquetas de git y convertirlas en notas.

  %p
    Los releases de Github crean un changelog no portable que sólo
    pueden ser mostrados a usuarios dentro del contexto de Github. Es
    posible hacer que luzcan muy parecidas al formato de <em>Mantenga un
    changelog</em>, pero tiende a ser un poco más complicado.

  %p
    La versión actual de los releases de Github es también
    discutiblemente no muy detectable por los usuarios finales,
    diferente a los típicos archivos en mayúsculas (<code>README</code>,
    <code>CONTRIBUTING</code>, etc.). Otro problema menor es que la
    interfaz actualmente no ofrece enlaces a los registros de los
    commits entre cada lanzamiento.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    ¿Se pueden analizar gramaticalmente los changelogs?

  %p
    Es difícil, porque las personas usan formatos y nombres de archivos
    muy distintos.

  %p
    #{link_to "Vandamme", data.links.vandamme} es una gema de Ruby creada por el
    equipo de Gemnasium que analiza
    gramaticalmente muchos (pero no todos) los changelogs de proyectos
    open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    ¿Qué hay sobre las versiones retiradas?

  %p
    «Yanked releases» son versiones que tuvieron que ser retiradas por
    un error grave o problema de seguridad. Con frecuencia estas
    versiones ni siquiera aparecen en los changelogs. Deberían. Así es
    como deberían mostrarse:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    La etiqueta <code>[YANKED]</code> está destacada por una razón: es
    importante que destaque. El hecho de estar entre corchetes la hace
    también más fácil de localizar programáticamente.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    ¿Deberías volver a escribir un changelog?

  %p
    Por supuesto. Siempre hay buenas razones para mejorar el changelog.
    A veces abro «pull requests» para añadir registros faltantes en el
    changelog de proyectos open source.

  %p
    También es posible que puedas descubrir que olvidaste señalar un
    cambio sin compatibilidad hacia atrás en las notas para una versión.
    En este caso es importante para ti actualizar el changelog.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    ¿Cómo puedo contribuir?

  %p
    Este documento no es la <strong>verdad absoluta</strong>; es mi
    cuidadosa opinión, junto con información y ejemplos que recopilé.

  %p
    Esto es porque quiero que la comunidad llegue a un consenso. Creo
    que la discusión es tan importante como el resultado final.

  %p
    Así que por favor <strong>#{link_to "comienza a colaborar", data.links.repo}
    </strong>.

.press
  %h3 Conversaciones
  %p
    Fui a #{link_to "The Changelog podcast", data.links.thechangelog} para hablar
    acerca del porqué los mantenedores y colaboradores deberían
    preocuparse por los changelogs, y también acerca de las motivaciones
    detrás de este proyecto.
```

## File: source/fa/1.0.0/index.html.haml
```haml
---
title: تاریخچهٔ تغییرات را نگه دارید
description: خروجی git log را به اسم تاریخچهٔ تغییرات پروژه منتشر نکنید.
language: fa
version: 1.0.0
---
<link type="text/css" rel="stylesheet" href="https://cdn.rawgit.com/rastikerdar/vazir-font/v19.0.0/dist/font-face.css">

<style>
body,html,h1,h2,h3,h4,h5,h6,a{font-family:Vazir;direction:rtl;text-align:right}
div.frequently-asked-questions h4:after{float:left}
pre {direction:ltr;text-align:left}
</style>

.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    تاریخچهٔ تغییرات چیست؟

  %p
    تاریخچهٔ تغییرات، فایلی شامل تغییرات عمدهٔ هر نسخه از پروژه به‌ترتیبِ زمان انتشار هر نسخه است.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    چرا باید لیست تغییرات را نگه‌داری کنیم؟

  %p
    برای اینکه کاربران بدانند در هر نسخه از پروژه چه اتفاقااتی افتاده است.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    لیست تغییرات به درد چه کسانی می‌خورد؟

  %p
    مصرف‌کننده‌ها، توسعه‌دهنده‌ها، کاربران نهایی و در یک کلام مردم.
    
  %p
    مسلماً وقتی نرم‌افزاری بروزرسانی می‌شود، کاربران آن دوست دارند بدانند چرا و چطور این تغییرات اعمال شده است.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    چطور یک تاریخچهٔ تغییرات خوب بسازیم؟

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    راهنمای کلی

  %ul
    %li
      تاریخچهٔ تغییرات  <em>برای انسان</em> است، نه ماشین.
    %li
      برای هر نسخه از پروژه، یک مدخل جدید تعریف کنید.
    %li
      تغییرات مشابه را دسته‌بندی کنید.
    %li
      نسخه‌ها و بخش‌ها قابل لینک‌شدن باشند.
    %li
      آخرین نسخه را در ابتدا قرار دهید.
    %li
      تاریخ انتشار هر نسخه را مشخص کنید.
    %li
      بگویید که آیا از #{link_to "نسخه‌بندی معنایی", data.links.semver} پیروی می‌کنید یا نه.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types انواع تغییرات

  %ul
    %li
      %code Added
      برای قابلیت‌های جدید.
    %li
      %code Changed
      برای تغییر در عملکردهای فعلی.
    %li
      %code Deprecated
      برای قابلیت‌هایی که حذف می‌شود.
    %li
      %code Removed
      برای قابلیت‌های حذف‌شده.
    %li
      %code Fixed
      برای خطاهای رفع‌شده.
    %li
      %code Security
      برای آسیب‌پذیری‌های امنیتی.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    چطور ثبت تغییرات پروژه را آسان‌تر کنیم؟

  %p
    در بالای فایل تاریخچهٔ تغییرات، بخشی را تحت عنوان <code>Unreleased</code> برای دنبال‌کردن تغییرات پیش‌رو در نظر بگیرید.

  %p این کار دو فایده دارد:

  %ul
    %li
      اولاً کاربران می‌توانند پیش از انتشار یک نسخهٔ جدید، از تغییراتی که می‌توانند در نسخهٔ بعدی انتظار داشته باشند مطلع شوند.
    %li
      دوماً هنگام انتشار نسخهٔ جدید، می‌توانید به‌راحتی بخش <code>Unreleased</code> را در قالب یک نسخهٔ جدید به بخش انتشاریافته‌ها منتقل کنید و دیگر لازم نباشد بروید ببینید چه تغییراتی ایجاد شده است.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    آیا تاریخچهٔ تغییرات می‌تواند بد نوشته شود؟
  
  %p بله. در شرایط زیر.:
  
  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    استفاده از خروجی log diff به عنوان تاریخچهٔ تغییرات

  %p
  استفاده از خروجی log diffها به عنوان تاریخچهٔ تغییرات، ایدهٔ بدی است. این خروجی مملو از اطلاعات درهم‌وبرهم است؛ مثلاً کامیت‌های مربوط به مرج، کامیت‌هایی با عناوین مبهم، تغییرات مربوط به مستندات و غیره.

  %p
    هدف از یک کامیت، ثبت یک گام انجام‌شده در روند تکامل سورس کد است. برخی از پروژه‌ها، کامیت‌های تمیزی دارند و برخی نه.

  %p
  هدف از یک مدخل جدید در فایل تاریخچهٔ تغییرات، مستندسازی شفاف تغییرات مهمِ پروژه برای کاربران نهایی است. هر مدخل در تاریخچهٔ تغییرات لزوماً متناظر با یک کامیت در تاریخچهٔ کامیت‌ها نیست. ممکن است چندین کامیت مختلف، مربوط به یک تغییر باشد و بالعکس یک کامیت (غیراستاندارد) دربردارندهٔ چندین تغییر مهم باشد.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    نادیده‌گرفتن چیزهای منسوخ‌شده
    
  %p
    وقتی کاربران از یک نسخه به نسخهٔ دیگر به‌روزرسانی می‌کنند، باید بدانند که چه موقع نسخهٔ جدید، تغییرات ناسازگار به‌همراه دارد؛ یعنی مثلاً بعد از بروزرسانی یک کتابخانه که پروژهٔ شما به آن وابسته بوده، آیا بخش‌هایی از پروژه از کار می‌افتد یا نه. اگر بله، کدام بخش‌ها. پاسخ به این سؤال کمک می‌کند توسعه‌دهنده‌ها متناسب با نسخهٔ جدید، قسمت‌های منسوخ‌شده را جایگزین کرده و بعد از آن اقدام به بروزرسانی کنند.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    تاریخ‌های گیج‌کننده

  %p
    در هر نقطه‌ای از دنیا شکل نمایش تاریخ متفاوت است؛ بنابراین پیداکردن یک قالب نمایش همه‌فهم برای کل مردم دنیا کار دشواری است. مزیت استفاده از قالب‌هایی مثل <code>17-07-2017</code> این است که در آن ترتیب بزرگترین به کوچکترین واحدها یعنی سال، ماه و روز رعایت شده است. این باعث می‌شود که برخلاف برخی از قالب‌ها که در آن جای ماه و روز عوض شده، درک تاریخ دچار ابهام نشود. با درنظرگرفتن این دلایل و این واقعیت که این فرمت، یک #{link_to "استاندارد ایزو", data.links.isodate} است، این قالب را برای مدخل‌های تاریخچهٔ تغییرات در نظر گرفته‌ایم.
  
  %aside
    اگر فکر می‌کنید جای موارد بیشتری در این لیست خالی است با
    = link_to "ارسال ایشو", data.links.issue
    یا ارسال پول‌ریکوئست به من کمک کنید.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    سوالات متداول
  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    آیا فرمت استانداردی برای تاریخچهٔ تغییرات وجود دارد؟

  %p
    حقیتاً نه. یک "راهنمای" #{link_to "راهنمای شیوه ثبت تاریخچهٔ تغییرات GNU", data.links.gnustyle}
    و یک #{link_to "فایل دو پاراگرافی بلندِ GNU NEWS", data.links.gnunews}
    وجود دارد که متأسفانه هردویشان نارسا و ناکافی هستند.

  %p
    این پروژه قصد دارد
    = link_to "قراردادی بهتر برای ثبت تاریخچهٔ تغییرات باشد", data.links.changelog
    که برگرفته از تجربیات جامعهٔ متن‌باز است.

  %p
    از نقد سالم، بحث‌وگفتگو و پیشنهادها
    = link_to "استقبال می‌کنیم.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    نام فایل تاریخچهٔ تغییرات چه چیزی باشد؟

  %p
    نام این فایل را <code>CHANGELOG.md</code> بگذارید. بعضی پروژه‌ها از
    <code>HISTORY</code>, <code>NEWS</code> یا <code>RELEASES</code> استفاده می‌کنند.

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    چرا از ریلیزهای گیت‌هاب استفاده نکنیم؟
  %p
    ابتکار خوبی است. در هنگام انتشار #{link_to "Release", data.links.github_releases}ها می‌توانید به جای عرضهٔ یک نسخهٔ ساده (مثلاً یک برچسب خالی با نام <code>v1.0.0</code>)، لیست تغییرات آن نسخه را در قالب یک یادداشت، به آن برچسب ضمیمه کنید.
  
  %p
    اما نکته این است که تاریخچهٔ تغییراتی که در هر ریلیز گیت‌هاب منتشر می‌کنید فقط در خودِ گیت‌هاب قابل استفاده است و به‌راحتی نمی‌توانید آن را به جای دیگری منتقل کنید. شاید بتوانید بسیار شبیه نسخهٔ پیشنهادی این سند کنید ولی کار زحمت‌داری است.
    
  %p
    مشاهدهٔ تغییرات پروژه در لابه‌لای ریلیزهای گیت‌هاب، به اندازهٔ دیدن همهٔ تغییرات در یک فایل، آسان و دلچسب نیست. فایل‌هایی با حرف بزرگ،
    (<code>README</code>, <code>CONTRIBUTING</code>, غیره.)
    جذابیت بیشتری برای کاربران دارد. مشکل کوچک بعدی این است که رابط کاربری فعلی اجازه نمی‌دهد بین تاریخچهٔ کامیت‌ها در عرضه‌های مختلف، پیوند برقرار کنید.
  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    آیا نمی‌توان مطابقت تاریخچهٔ تغییرات با موازین این سند را به صورت خودکار انجام داد؟

  %p
    مشکل است؛ چون افراد از فرمت‌ها و نام‌گذاری‌های متعددی برای این فایل استفاده می‌کنند.

  %p
    #{link_to "Vandamme", data.links.vandamme} یک روبی gem است که توسط تیم
    Gemnasium ساخته شده است. این ابزار، تاریخچهٔ تغییرات پروژه‌های متن‌باز (البته نه همه) را بررسی می‌کند.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    عرضه‌های yanked چطور؟

  %p
    عرضه‌های Yanked نسخه‌هایی هستند که به خاطر باگ جدی یا مشکل امنیتی باید گرفته شوند. این نسخه‌ها اغلب حتی در تاریخچهٔ تغییرات نیز دیده نمی‌شوند، اما باید اضافه شوند. آن‌ها را به صورت زیر در فایل تاریخچهٔ تغییرات نشان دهید:
  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    برچسب <code>[YANKED]</code> را به این دلیل، به‌شکل متمایز در تاریخچهٔ تغییرات آورده‌ایم که نظرها را به خود جلب کند. بعلاوه وجود کروشه، کار پارزرها را راحت‌تر می‌کند.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    آیا لازم است تاریخچهٔ تغییرات را بازنویسی کنیم؟

  %p
    قطعاً. معمولاً همیشه دلیل خوبی برای بهبود تاریخچهٔ تغییرات وجود دارد. من معمولاً به‌طور منظم، فایل تاریخچهٔ تغییرات را برای پروژه‌های متن‌بازی که این کار را انجام نداده‌اند می‌نویسم و تغییرات را به صورت یک پول‌ریکوئست ارائه می‌دهم.

  %p
    گاهی نیز ممکن است یادتان رفته باشد که تغییرات ناسازگارِ یک نسخه را ثبت کنید. در چنین شرایطی نیز واضح است که باید تاریخچهٔ تغییرات را بروزرسانی کنید.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    چطور می‌توانم مشارکت کنم؟

  %p
    این سند، <strong>کامل و بی‌نقص</strong> نیست؛ صرفاً نظرات به‌دقت بررسی‌شده‌ای است که همراه اطلاعات و مثال‌هایی، در اینجا قرار داده‌ام.

  %p
    این به خاطر آن است که می‌خواهم جامعهٔ ما توسعه‌دهنده‌ها به یک اتفاق‌نظر برسد. من معتقدم بحث‌وگفتگو به‌اندازهٔ نتیجه مهم است.

  %p
    بنابراین شما نیز <strong>#{link_to "دست به کار شوید", data.links.repo}</strong>.

.press
  %h3 گفتگو
  %p
    در #{link_to "پادکست Changelog", data.links.thechangelog}، گفتگویی داشته‌ام دربارهٔ انگیزهٔ پشت این پروژه و اینکه چرا متصدیان نگهداری و مشارکت‌کنندگان پروژه‌ها باید به تاریخچهٔ تغییرات اهمیت بدهند.
```

## File: source/fa/1.1.0/index.html.haml
```haml
---
title: تاریخچهٔ تغییرات را نگه دارید
description: خروجی git log را به اسم تاریخچهٔ تغییرات پروژه منتشر نکنید
language: fa
version: 1.1.0
---
<link type="text/css" rel="stylesheet" href="https://cdn.rawgit.com/rastikerdar/vazir-font/v19.0.0/dist/font-face.css">

<style>
body,html,h1,h2,h3,h4,h5,h6,a{font-family:Vazir;direction:rtl;text-align:right}
div.frequently-asked-questions h4:after{float:left}
pre {direction:ltr;text-align:left}
</style>

.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    تاریخچهٔ تغییرات چیست؟

  %p
    تاریخچهٔ تغییرات، فایلی شامل تغییرات عمدهٔ هر نسخه از پروژه به‌ترتیبِ زمان انتشار هر نسخه است.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    چرا باید لیست تغییرات را نگه‌داری کنیم؟

  %p
    برای اینکه کاربران بدانند در هر نسخه از پروژه چه اتفاقااتی افتاده است.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    لیست تغییرات به درد چه کسانی می‌خورد؟

  %p
    مصرف‌کننده‌ها، توسعه‌دهنده‌ها، کاربران نهایی و در یک کلام مردم.
    
  %p
    مسلماً وقتی نرم‌افزاری بروزرسانی می‌شود، کاربران آن دوست دارند بدانند چرا و چطور این تغییرات اعمال شده است.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    چطور یک تاریخچهٔ تغییرات خوب بسازیم؟

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    راهنمای کلی

  %ul
    %li
      تاریخچهٔ تغییرات  <em>برای انسان</em> است، نه ماشین.
    %li
      برای هر نسخه از پروژه، یک مدخل جدید تعریف کنید.
    %li
      تغییرات مشابه را دسته‌بندی کنید.
    %li
      نسخه‌ها و بخش‌ها قابل لینک‌شدن باشند.
    %li
      آخرین نسخه را در ابتدا قرار دهید.
    %li
      تاریخ انتشار هر نسخه را مشخص کنید.
    %li
      بگویید که آیا از #{link_to "نسخه‌بندی معنایی", data.links.semver} پیروی می‌کنید یا نه.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types انواع تغییرات

  %ul
    %li
      %code Added
      برای قابلیت‌های جدید.
    %li
      %code Changed
      برای تغییر در عملکردهای فعلی.
    %li
      %code Deprecated
      برای قابلیت‌هایی که حذف می‌شود.
    %li
      %code Removed
      برای قابلیت‌های حذف‌شده.
    %li
      %code Fixed
      برای خطاهای رفع‌شده.
    %li
      %code Security
      برای آسیب‌پذیری‌های امنیتی.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    چطور ثبت تغییرات پروژه را آسان‌تر کنیم؟

  %p
    در بالای فایل تاریخچهٔ تغییرات، بخشی را تحت عنوان <code>Unreleased</code> برای دنبال‌کردن تغییرات پیش‌رو در نظر بگیرید.

  %p این کار دو فایده دارد:

  %ul
    %li
      اولاً کاربران می‌توانند پیش از انتشار یک نسخهٔ جدید، از تغییراتی که می‌توانند در نسخهٔ بعدی انتظار داشته باشند مطلع شوند.
    %li
      دوماً هنگام انتشار نسخهٔ جدید، می‌توانید به‌راحتی بخش <code>Unreleased</code> را در قالب یک نسخهٔ جدید به بخش انتشاریافته‌ها منتقل کنید و دیگر لازم نباشد بروید ببینید چه تغییراتی ایجاد شده است.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    آیا تاریخچهٔ تغییرات می‌تواند بد نوشته شود؟
  
  %p بله. در شرایط زیر.:
  
  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    استفاده از خروجی log diff به عنوان تاریخچهٔ تغییرات

  %p
    استفاده از خروجی log diffها به عنوان تاریخچهٔ تغییرات، ایدهٔ بدی است. این خروجی مملو از اطلاعات درهم‌وبرهم است؛ مثلاً کامیت‌های مربوط به مرج، کامیت‌هایی با عناوین مبهم، تغییرات مربوط به مستندات و غیره.

  %p
    هدف از یک کامیت، ثبت یک گام انجام‌شده در روند تکامل سورس کد است. برخی از پروژه‌ها، کامیت‌های تمیزی دارند و برخی نه.

  %p
    هدف از یک مدخل جدید در فایل تاریخچهٔ تغییرات، مستندسازی شفاف تغییرات مهمِ پروژه برای کاربران نهایی است. هر مدخل در تاریخچهٔ تغییرات لزوماً متناظر با یک کامیت در تاریخچهٔ کامیت‌ها نیست. ممکن است چندین کامیت مختلف، مربوط به یک تغییر باشد و بالعکس یک کامیت (غیراستاندارد) دربردارندهٔ چندین تغییر مهم باشد.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    نادیده‌گرفتن چیزهای منسوخ‌شده
    
  %p
    وقتی کاربران از یک نسخه به نسخهٔ دیگر به‌روزرسانی می‌کنند، باید بدانند که چه موقع نسخهٔ جدید، تغییرات ناسازگار به‌همراه دارد؛ یعنی مثلاً بعد از بروزرسانی یک کتابخانه که پروژهٔ شما به آن وابسته بوده، آیا بخش‌هایی از پروژه از کار می‌افتد یا نه. اگر بله، کدام بخش‌ها. پاسخ به این سؤال کمک می‌کند توسعه‌دهنده‌ها متناسب با نسخهٔ جدید، قسمت‌های منسوخ‌شده را جایگزین کرده و بعد از آن اقدام به بروزرسانی کنند.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    تاریخ‌های گیج‌کننده

  %p
    در هر نقطه‌ای از دنیا شکل نمایش تاریخ متفاوت است؛ بنابراین پیداکردن یک قالب نمایش همه‌فهم برای کل مردم دنیا کار دشواری است. مزیت استفاده از قالب‌هایی مثل <code>17-07-2017</code> این است که در آن ترتیب بزرگترین به کوچکترین واحدها یعنی سال، ماه و روز رعایت شده است. این باعث می‌شود که برخلاف برخی از قالب‌ها که در آن جای ماه و روز عوض شده، درک تاریخ دچار ابهام نشود. با درنظرگرفتن این دلایل و این واقعیت که این فرمت، یک #{link_to "استاندارد ایزو", data.links.isodate} است، این قالب را برای مدخل‌های تاریخچهٔ تغییرات در نظر گرفته‌ایم.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    ناهمگونی ثبت تغییرات
    
  %p	
    تاریخچهٔ تغییراتی که فقط شامل برخی از تغییرات است می‌تواند به اندازهٔ نبود تاریخچهٔ تغییرات خطرناک باشد. با اینکه لزومی به ثبت همهٔ تغییرات مثلاً حذف یک فاصلهٔ خالی نیست، باید هر تغییر مهمی را در تاریخچهٔ تغییرات ثبت کرد. نایکدستی در ثبت تغییرات ممکن است اعتماد کاربران را به تاریخچهٔ تغییرات به عنوان یگانه مرجع مشاهدهٔ تغییرات سلب کند.  یک تاریخچهٔ تغییرات خوب باید به‌طور منظم بروزرسانی شود. 
  
  %aside
    اگر فکر می‌کنید جای موارد بیشتری در این لیست خالی است با
    = link_to "ارسال ایشو", data.links.issue
    یا ارسال پول‌ریکوئست به من کمک کنید.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    سوالات متداول
  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    آیا فرمت استانداردی برای تاریخچهٔ تغییرات وجود دارد؟

  %p
    حقیتاً نه. یک "راهنمای" #{link_to "راهنمای شیوه ثبت تاریخچهٔ تغییرات GNU", data.links.gnustyle}
    و یک #{link_to "فایل دو پاراگرافی بلندِ GNU NEWS", data.links.gnunews}
    وجود دارد که متأسفانه هردویشان نارسا و ناکافی هستند.

  %p
    این پروژه قصد دارد
    = link_to "قراردادی بهتر برای ثبت تاریخچهٔ تغییرات باشد", data.links.changelog
    که برگرفته از تجربیات جامعهٔ متن‌باز است.

  %p
    از نقد سالم، بحث‌وگفتگو و پیشنهادها
    = link_to "استقبال می‌کنیم.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    نام فایل تاریخچهٔ تغییرات چه چیزی باشد؟

  %p
    نام این فایل را <code>CHANGELOG.md</code> بگذارید. بعضی پروژه‌ها از
    <code>HISTORY</code>, <code>NEWS</code> یا <code>RELEASES</code> استفاده می‌کنند.

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    چرا از ریلیزهای گیت‌هاب استفاده نکنیم؟
  %p
    ابتکار خوبی است. در هنگام انتشار #{link_to "Release", data.links.github_releases}ها می‌توانید به جای عرضهٔ یک نسخهٔ ساده (مثلاً یک برچسب خالی با نام <code>v1.0.0</code>)، لیست تغییرات آن نسخه را در قالب یک یادداشت، به آن برچسب ضمیمه کنید.
  
  %p
    اما نکته این است که تاریخچهٔ تغییراتی که در هر ریلیز گیت‌هاب منتشر می‌کنید فقط در خودِ گیت‌هاب قابل استفاده است و به‌راحتی نمی‌توانید آن را به جای دیگری منتقل کنید. شاید بتوانید بسیار شبیه نسخهٔ پیشنهادی این سند کنید ولی کار زحمت‌داری است.
    
  %p
    مشاهدهٔ تغییرات پروژه در لابه‌لای ریلیزهای گیت‌هاب، به اندازهٔ دیدن همهٔ تغییرات در یک فایل، آسان و دلچسب نیست. فایل‌هایی با حرف بزرگ،
    (<code>README</code>, <code>CONTRIBUTING</code>, غیره.)
    جذابیت بیشتری برای کاربران دارد. مشکل کوچک بعدی این است که رابط کاربری فعلی اجازه نمی‌دهد بین تاریخچهٔ کامیت‌ها در عرضه‌های مختلف، پیوند برقرار کنید.
  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    آیا نمی‌توان مطابقت تاریخچهٔ تغییرات با موازین این سند را به صورت خودکار انجام داد؟

  %p
    مشکل است؛ چون افراد از فرمت‌ها و نام‌گذاری‌های متعددی برای این فایل استفاده می‌کنند.

  %p
    #{link_to "Vandamme", data.links.vandamme} یک روبی gem است که توسط تیم
    Gemnasium ساخته شده است. این ابزار، تاریخچهٔ تغییرات پروژه‌های متن‌باز (البته نه همه) را بررسی می‌کند.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    عرضه‌های yanked چطور؟

  %p
    عرضه‌های Yanked نسخه‌هایی هستند که به خاطر باگ جدی یا مشکل امنیتی باید گرفته شوند. این نسخه‌ها اغلب حتی در تاریخچهٔ تغییرات نیز دیده نمی‌شوند، اما باید اضافه شوند. آن‌ها را به صورت زیر در فایل تاریخچهٔ تغییرات نشان دهید:
  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    برچسب <code>[YANKED]</code> را به این دلیل، به‌شکل متمایز در تاریخچهٔ تغییرات آورده‌ایم که نظرها را به خود جلب کند. بعلاوه وجود کروشه، کار پارزرها را راحت‌تر می‌کند.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    آیا لازم است تاریخچهٔ تغییرات را بازنویسی کنیم؟

  %p
    قطعاً. معمولاً همیشه دلیل خوبی برای بهبود تاریخچهٔ تغییرات وجود دارد. من معمولاً به‌طور منظم، فایل تاریخچهٔ تغییرات را برای پروژه‌های متن‌بازی که این کار را انجام نداده‌اند می‌نویسم و تغییرات را به صورت یک پول‌ریکوئست ارائه می‌دهم.

  %p
    گاهی نیز ممکن است یادتان رفته باشد که تغییرات ناسازگارِ یک نسخه را ثبت کنید. در چنین شرایطی نیز واضح است که باید تاریخچهٔ تغییرات را بروزرسانی کنید.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    چطور می‌توانم مشارکت کنم؟

  %p
    این سند، <strong>کامل و بی‌نقص</strong> نیست؛ صرفاً نظرات به‌دقت بررسی‌شده‌ای است که همراه اطلاعات و مثال‌هایی، در اینجا قرار داده‌ام.

  %p
    این به خاطر آن است که می‌خواهم جامعهٔ ما توسعه‌دهنده‌ها به یک اتفاق‌نظر برسد. من معتقدم بحث‌وگفتگو به‌اندازهٔ نتیجه مهم است.

  %p
    بنابراین شما نیز <strong>#{link_to "دست به کار شوید", data.links.repo}</strong>.

.press
  %h3 گفتگو
  %p
    در #{link_to "پادکست Changelog", data.links.thechangelog}، گفتگویی داشته‌ام دربارهٔ انگیزهٔ پشت این پروژه و اینکه چرا متصدیان نگهداری و مشارکت‌کنندگان پروژه‌ها باید به تاریخچهٔ تغییرات اهمیت بدهند.
```

## File: source/fr/0.3.0/index.html.haml
```haml
---
title: Tenez un Changelog
description: Ne laissez pas vos amis utiliser les logs git comme CHANGELOGs
language: fr
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Qu’est-ce qu’un journal des modifications (change log) ?
  Un journal des modifications (ou change log) est un fichier qui contient
  une liste triée chronologiquement des changements notables pour chaque
  version d’un projet

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Quel est l’intérêt d’un change log ?
  Il permet aux utilisateurs et aux contributeurs de voir plus simplement
  les changements notables qui ont été réalisés entre chaque publication
  (ou version) du projet.

  ### Pourquoi devrais-je m’en préoccuper ?
  Parce que les logiciels sont pour les gens. Si vous ne vous en souciez pas,
  pourquoi contribuez-vous à l’open source ? Il doit sûrement y avoir un
  "kernel" (ha!) d’intérêt pour ça quelque part dans votre cher petit
  cerveau.

  J’ai [discuté avec Adam Stacoviak et Jerod Santo dans le podcast
  "The Changelog"][thechangelog] (approprié, non ?) des raisons pour
  lesquelles les mainteneurs et les contributeurs devraient s’en préoccuper ;
  également des motivations derrière ce projet.
  Si vous avez le temps (1:06), l’écoute vaut le coup.

  ### Qu’est-ce qui fait un bon change log ?
  Heureux de vous entendre le demander.

  Un bon change log se conforme à ces principes:

  - Il est fait pour des humains, pas des machines, la lisibilité est donc
    cruciale.
  - Facilité de faire un lien vers n’importe quelle section (d’où le Markdown
    plutôt que le text brut).
  - Une sous-section par version.
  - Liste les publications dans l’ordre chronologique inverse (les plus récentes
    en haut).
  - Toutes les dates sont au format `AAAA-MM-JJ`. (Exemple: `2012-06-02` pour le
    `2 Juin 2012`.) C’est international, [raisonnable](https://xkcd.com/1179/) et
    indépendant de la langue.
  - Mentionne explicitement si le projet respecte le
    [Versionnage Sémantique][semver].
  - Chaque version devrait:
    - Lister sa date de publication au format ci-dessus.
    - Regrouper les changements selon leur impact sur le projet, comme suit:
      - `Added` pour les nouvelles fonctionnalités.
      - `Changed` pour les changements au sein des fonctionnalités déjà
        existantes.
      - `Deprecated` pour les fonctionnalités qui seront supprimées dans la
        prochaine publication.
      - `Removed` pour les anciennes fonctionnalités `Deprecated` qui viennent
        d’être supprimées.
      - `Fixed` pour les corrections de bugs.
      - `Security` pour encourager les utilisateurs à mettre à niveau afin
        d’éviter des failles de sécurité.

  ### Comment puis-je minimiser le travail nécessaire ?
  Ayez toujours une section `"Unreleased"` en haut du fichier afin de lister
  tous les changements pas encore publiés.

  Cela rempli deux objectifs:

  - Les gens peuvent voir les changements auxquels ils peuvent s’attendrent dans
    les prochaines publications
  - Au moment de la publication, vous n’avez qu’à remplacer `"Unreleased"` par
    la nouvelle version et rajouter une nouvelle section `"Unreleased"`
    au-dessus.

  ### Quelles sont les choses qui rendent tristes les licornes ?
  Très bien… parlons-en.

  - **Recopier les dernier logs git.** Ne faites simplement pas cela, vous
    n’aidez personne.
  - **Ne pas mettre l’accent sur les fonctionnalités dépréciées.** Quand les
    gens mettent à niveau d’une version vers une autre, le fait que quelque
    chose ne soit pas compatible devrait être évident.
  - **Les dates dans des formats spécifiques à une région.** Aux États-Unis, les
    gens mettent le mois en premier ("06-02-2012" pour le 2 Juin 2012, ce qui
    n’a *pas* de sens), tandis que beaucoup de gens dans le reste du monde
    l’écrivent de la façon robotique suivante "2 Juin 2012", alors qu’ils le
    prononcent différement. "2012-06-02" fonctionne logiquement, du plus grand
    au plus petit, ne laisse pas place à la confusion avec un autre format et
    est un [standard ISO](http://www.iso.org/iso/home/standards/iso8601.htm).
    Voilà pourquoi il est le format de dates recommandé pour les change logs.

  Il y en a d’autres. Aidez-moi à collecter ces larmes de licornes en
  [ouvrant une issue][issues] ou une pull request.

  ### Y a-t-il un format de change log standard ?
  Malheureusement, non. Du calme. Je sais que vous êtes en train de vous
  précipiter afin de trouver le lien vers le guide pour change logs GNU, ou le
  fichier GNU NEWS "guideline" de deux paragraphes. Le guide GNU est un bon
  début mais il est malheureusement simpliste. Il n'y a rien de mal avec le fait
  d'être simpliste, mais quand les gens ont besoin d'être guidés, ce n'est que
  rarement utile. Spécialement quand il a beaucoup de situations et de cas
  particuliers à prendre en compte.

  Ce projet [contient ce que j'espère devenir une meilleur convention pour les fichiers CHANGELOG][CHANGELOG].
  Je ne pense pas que le status quo soit suffisant et je pense qu'en tant que
  communauté, nous pouvons arriver à de meilleures conventions si nous essayons
  d'extraire les meilleures pratiques depuis les vrais projets logiciels. S'il
  vous plaît, jetez-y un coup d'oeil et rappelez vous que les
  [discussions et les suggestions d'améliorations sont les bienvenues][issues]!

  ### Comment doit-être nommé le fichier de change log ?
  Eh bien, si l'exemple ci-dessus ne vous suffit pas à le deviner,
  `CHANGELOG.md` est la meilleure convention à ce jour.

  Certains projets utilisent aussi `HISTORY.txt`, `HISTORY.md`, `History.md`,
  `NEWS.txt`, `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`,
  `releases.md`, etc.

  C'est un véritable bazar. Tous ces noms ne font que rendre plus difficile son
  repérage par les gens.

  ### Pourquoi les gens ne recopient pas simplement les derniers logs git ?
  Parce que les logs gits sont remplis de bruit - par définition. Ils ne peuvent
  pas faire office de change log convenable, même dans un hypothétique projet
  tenu par des humains parfaits qui ne font jamais la moindre faute de frappe,
  n'oublient jamais de committer les nouveaux fichiers, ne manquent jamais
  aucune partie d'un refactoring. Le but d'un commit est de documenter une étape
  atomique dans le processus par lequel le code évolue d'un état à un autre. Le
  but d'un change log est de documenter les différences notables entre ces
  états.

  La différence entre des bons commentaires et le code lui-même est la même que
  celle entre un change log et les journaux git: l'un décrit le *pourquoi*,
  l'autre le comment.

  ### Les change logs peuvent-ils être parsés automatiquement ?
  C'est difficile, parce que les gens suivent des formats et utilisent des noms
  de fichiers très différents.

  [Vandamme][vandamme] est une Ruby gem
  créée par l'équipe [Gemnasium][gemnasium] qui parse les change logs de
  beaucoup (mais pas tous) de projets open source.

  ### Pourquoi cette alternance entre les graphies "CHANGELOG" et "change log" ?
  "CHANGELOG" est le nom pour le fichier même. C'est un peu imposant, mais dû à
  une convention historique suivie par beaucoup de projets open source. Il
  existe d'autres fichiers similaires, par exemple : [`README`][README],
  [`LICENSE`][LICENSE], and [`CONTRIBUTING`][CONTRIBUTING].

  L'écriture en majuscule (qui dans les vieux systèmes d'exploitation faisait
  apparaître ces fichiers en haut) de ces noms de fichiers est utilisée pour
  attirer l'attention sur eux. Puisqu'ils font partie des méta-données
  importantes du projet, ils pourraient être utiles à quiconque voulant y
  l'utiliser ou y contribuer, tout comme les
  [badges de projet open source][shields].

  Quand j'utilise la graphie "change log", je fais référence à la fonction de ce
  fichier: journaliser les changements.

  ### Qu'en est-il des publications "yanked" ?
  Les publications yanked sont des version qui ont dû être retirées du fait d'un
  sérieux bug ou d'un problème de sécurité. Souvent ces version n'apparaissent
  même pas dans les change logs. Elles devraient. Voilà comment les afficher:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Le tag `[YANKED]` n'est pas mis en avant pour rien. C'est important pour les
  gens de le remarquer. Puisqu'il est entouré par des crochets, c'est aussi plus
  facile à parser pour un programme.

  ### Devriez-vous réécrire un change log ?
  Bien sûr. Il y a toujours de bonnes raisons d'améliorer un change log.
  J'ouvre souvent des pull requests pour ajouter des publications manquantes sur
  des projets open source avec des change logs non-maintenus.

  Il est aussi possible que vous découvriez que vous aviez oublié de mentionner
  un changement majeur dans les notes de version. Il est évidemment important
  que vous mettiez votre change log à jour dans ce cas.

  ### Comment puis-je contribuer ?
  Ce document n'est pas la **vérité**; c'est mon opinion soigneusement
  réfléchie, accompagnée d'informations et d'exemples que j'ai rassemblés.
  Bien que je fournisse un véritable [CHANGELOG][] sur [le dépôt GitHub][gh],
  je n'ai volontairement pas créé de véritable *publication* ou de liste précise
  de règles à suivre (comme le fait [SemVer.org][semver], par exemple).

  C'est parce que je veux que notre communauté atteigne un consensus. Je crois
  que la discussion est aussi importante que le résultat final.

  Donc s'il vous plaît, [**mettez-vous y**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/fr/1.0.0/index.html.haml
```haml
---
title: Tenez un Changelog
description: Ne laissez pas vos amis utiliser les logs git comme changelogs
language: fr
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Qu'est-ce qu'un changelog ?

  %p
    Un changelog (journal des modifications) est un fichier qui contient
    une liste triée chronologiquement des changements notables pour
    chaque version d’un projet

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Pourquoi tenir un changelog ?

  %p
    Pour permettre aux utilisateurs et contributeurs de voir précisément
    quels changements notables ont été faits entre chaque publication
    (ou version) d'un projet.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Qui a besoin d'un changelog ?

  %p
    Tout le monde. Qu'ils soient consommateurs ou développeurs, les
    utilisateurs de logiciels sont des êtres humains qui se soucient
    de connaître le contenu des logiciels qu'ils utilisent. Quand un
    logiciel change, ces mêmes personnes veulent savoir comment et pourquoi.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Comment créer un bon changelog ?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Principes Directeurs

  %ul
    %li
      Les changelogs sont <em>pour les êtres humains</em>, pas les machines.
    %li
      Il doit y avoir une section pour chaque version.
    %li
      Les changements similaires doivent être groupés.
    %li
      Les versions et sections doivent être liables.
    %li
      La version la plus récente se situe en haut du fichier.
    %li
      La date de publication de chaque version est indiquée.
    %li
      Indiquer si le projet respecte le #{link_to "Versionnage Sémantique", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Types de changements

  %ul
    %li
      %code Added
      pour les nouvelles fonctionnalités.
    %li
      %code Changed
      pour les changements aux fonctionnalités préexistantes.
    %li
      %code Deprecated
      pour les fonctionnalités qui seront bientôt supprimées.
    %li
      %code Removed
      pour les fonctionnalités désormais supprimées.
    %li
      %code Fixed
      pour les corrections de bugs.
    %li
      %code Security
      en cas de vulnérabilités.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Comment puis-je minimiser le travail nécessaire pour maintenir un changelog ?

  %p
    Gardez une section <code>Unreleased</code> en haut du fichier afin de lister
    tous les changements qui n'ont pas encore été publiés.

  %p Cela permet deux choses:

  %ul
    %li
      Les gens peuvent voir les changements auxquels ils peuvent s’attendre
      dans les prochaines publications.
    %li
      Au moment de la publication, vous pouvez déplacer les changements listés
      dans la section <code>Unreleased</code> dans la section d'une nouvelle
      version.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Est-ce que les changelogs peuvent être mauvais ?

  %p Oui. Voici quelques exemples de changelogs plutôt inutiles.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffs de journaux gits

  %p
    Utiliser des diffs de journaux gits en tant que changelogs est une mauvaise
    idée: ils sont remplis de bruit. Les journaux gits sont remplis de choses
    comme des commits de merge, des commits avec des titres obscurs, des
    changements concernant la documentation, etc.

  %p
    Le but d'un commit est de documenter une étape dans l'évolution du code
    source. Certains projets nettoient leurs commits, d'autres non.

  %p
    Le but d'une section de changelog est de documenter les différences
    notables entre deux versions, souvent à travers de multiples
    commits, afin de les communiquer clairement aux utilisateurs.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorer les dépréciations

  %p
    Lorsque l'on passe d'une version d'un logiciel à une autre, il devrait être
    très clair si quelque chose peut ne plus fonctionner. Il devrait être
    possible de mettre à jour vers une version qui offre une liste des
    fonctionnalités dépréciées, de retirer ce qui est déprécié, puis de mettre
    à jour vers la version où les dépréciations deviennent des suppressions de
    fonctionnalités.

  %p
    Si vous ne faites rien d'autre, listez les dépréciations, les supressions de
    fonctionnalités, et tous les changements non rétrocompatibles dans votre
    changelog.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Dates confuses

  %p
    Aux États-Unis, les gens mettent le mois en premier (<code>06-02-2012</code>
    pour le 2 juin 2012), tandis que beaucoup de gens dans le reste du monde
    l’écrivent de la façon robotique suivante « 2 Juin 2012 », alors qu’ils le
    prononcent différement. <code>2012-06-02</code> fonctionne logiquement, du plus grand
    au plus petit, ne laisse pas place à la confusion avec un autre format et
    est un #{link_to "standard ISO", data.links.isodate}. C'est pour cela que ce format de date
    est recommandé pour les changelogs.

  %aside
    Il y en a d’autres. Aidez-moi à collecter ces antipatrons en
    #{link_to "ouvrant une issue", data.links.issue} ou une pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Questions fréquemment posées

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Y a-t-il un format de changelog standard ?

  %p
    Pas vraiment. Il y a le guide de style GNU pour les changelogs, ou les
    instructions GNU de deux paragraphes pour les fichiers NEWS. Ces deux
    ressources sont inappropriées et insuffisantes.

  %p
    Ce projet vise à devenir
    = link_to "une meilleure convention pour les changelogs.", data.links.changelog
    Il a pour origine l'observation et le rassemblement des bonne pratiques dans
    la communauté open source.

  %p
    Les critiques constructives, discussions et suggestions pour améliorer ce
    projet #{link_to "sont les bienvenues.", data.links.issue}.


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Comment doit-être nommé le fichier de changelog ?

  %p
    Nommez-le <code>CHANGELOG.md</code>. Certains projets utilisent
    <code>HISTORY</code>, <code>NEWS</code> ou <code>RELEASES</code>.

  %p
    Même s'il est facile d'imaginer que le nom d'un fichier de changelog n'a
    pas grand intérêt, pourquoi rendre la tâche plus difficile que nécessaire
    aux utilisateurs qui cherchent à trouver les changements notables de votre
    projet ?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Quid de GitHub Releases ?

  %p
    C'est une excellente initiative. #{link_to "Releases", data.links.github_releases} peut être
    utilisé pour transformer de simples tags git (par exemple un tag nommé
    <code>v1.0.0</code>) en notes de publication riches soit en ajoutant
    manuellement des notes soit en utilisant les messages de tags gits annotés
    et en les transformant en notes.

  %p
    GitHub Releases crée un changelog non-portable qui n'est visible par les
    utilisateurs qu'à l'intérieur du contexte de GitHub. Il est possible de
    suivre le même format que Keep a Changelog, mais c'est plus difficile.

  %p
    La version actuelle de GitHub Releases est également difficile à découvrir
    pour les utilisateurs, contrairement aux fichiers en majuscules typiques
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.). Un autre souci
    mineur est que l'interface n'offre actuellement pas de lien vers les
    journaux gits entre chaque publication.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Les changelogs peuvent-ils être parsés automatiquement ?

  %p
    C'est difficile, parce que les gens suivent des formats et utilisent des noms
    de fichiers très différents.

  %p
    #{link_to "Vandamme", data.links.vandamme} est une Ruby gem créée par l'équipe
    Gemnasium qui parse les changelogs de
    beaucoup (mais pas tous) de projets open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Qu'en est-il des publications « yanked » (retirées) ?

  %p
    Les publications yanked sont des version qui ont dû être retirées du fait d'un
    sérieux bug ou d'un problème de sécurité. Souvent ces version n'apparaissent
    même pas dans les changelogs. Elles devraient. Voilà comment les afficher :

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Le tag <code>[YANKED]</code> n'est pas mis en avant pour rien. Il est
    important qu'il soit remarqué. Il est également plus facile à digérer pour
    un programme puisqu'il est entouré par des crochets.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Devriez-vous réécrire un changelog ?

  %p
    Bien sûr. Il y a toujours de bonnes raisons d'améliorer un changelog.
    J'ouvre souvent des pull requests pour ajouter des publications manquantes sur
    des projets open source avec des changelogs non-maintenus.

  %p
    Il est aussi possible que vous découvriez que vous aviez oublié de mentionner
    un changement majeur dans les notes de version. Il est évidemment important
    que vous mettiez votre changelog à jour dans ce cas.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Comment puis-je contribuer ?

  %p
    Ce document n'est pas la **vérité**; c'est mon opinion soigneusement
    réfléchie, accompagnée d'informations et d'exemples que j'ai rassemblés.

  %p
    C'est parce que je veux que notre communauté atteigne un consensus. Je crois
    que la discussion est aussi importante que le résultat final.

  %p
    Donc s'il vous plaît, <strong>#{link_to "participez", data.links.repo}</strong>.

.press
  %h3 Conversations
  %p
    J'ai participé au podcast #{link_to "The Changelog", data.links.thechangelog}
    pour expliquer pourquoi les mainteneurs et contributeurs doivent se
    soucier des changelogs et également des motivations derrière ce projet.
```

## File: source/fr/1.1.0/index.html.haml
```haml
---
title: Tenez un Changelog
description: Ne laissez pas vos amis utiliser les logs git comme changelogs
language: fr
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Qu'est-ce qu'un changelog ?

  %p
    Un changelog (journal des modifications) est un fichier qui contient
    une liste triée chronologiquement des changements notables pour
    chaque version d’un projet

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Pourquoi tenir un changelog ?

  %p
    Pour permettre aux utilisateurs et contributeurs de voir précisément
    quels changements notables ont été faits entre chaque publication
    (ou version) d'un projet.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Qui a besoin d'un changelog ?

  %p
    Tout le monde. Qu'ils soient consommateurs ou développeurs, les
    utilisateurs de logiciels sont des êtres humains qui se soucient
    de connaître le contenu des logiciels qu'ils utilisent. Quand un
    logiciel change, ces mêmes personnes veulent savoir comment et pourquoi.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Comment créer un bon changelog ?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Principes Directeurs

  %ul
    %li
      Les changelogs sont <em>pour les êtres humains</em>, pas les machines.
    %li
      Il doit y avoir une section pour chaque version.
    %li
      Les changements similaires doivent être groupés.
    %li
      Les versions et sections doivent être liables.
    %li
      La version la plus récente se situe en haut du fichier.
    %li
      La date de publication de chaque version est indiquée.
    %li
      Indiquer si le projet respecte le #{link_to "Versionnage Sémantique", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Types de changements

  %ul
    %li
      %code Added
      pour les nouvelles fonctionnalités.
    %li
      %code Changed
      pour les changements aux fonctionnalités préexistantes.
    %li
      %code Deprecated
      pour les fonctionnalités qui seront bientôt supprimées.
    %li
      %code Removed
      pour les fonctionnalités désormais supprimées.
    %li
      %code Fixed
      pour les corrections de bugs.
    %li
      %code Security
      en cas de vulnérabilités.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Comment puis-je minimiser le travail nécessaire pour maintenir un changelog ?

  %p
    Gardez une section <code>Unreleased</code> en haut du fichier afin de lister
    tous les changements qui n'ont pas encore été publiés.

  %p Cela permet deux choses:

  %ul
    %li
      Les gens peuvent voir les changements auxquels ils peuvent s’attendre
      dans les prochaines publications.
    %li
      Au moment de la publication, vous pouvez déplacer les changements listés
      dans la section <code>Unreleased</code> dans la section d'une nouvelle
      version.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Est-ce que les changelogs peuvent être mauvais ?

  %p Oui. Voici quelques exemples de changelogs plutôt inutiles.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffs de journaux gits

  %p
    Utiliser des diffs de journaux gits en tant que changelogs est une mauvaise
    idée: ils sont remplis de bruit. Les journaux gits sont remplis de choses
    comme des commits de merge, des commits avec des titres obscurs, des
    changements concernant la documentation, etc.

  %p
    Le but d'un commit est de documenter une étape dans l'évolution du code
    source. Certains projets nettoient leurs commits, d'autres non.

  %p
    Le but d'une section de changelog est de documenter les différences
    notables entre deux versions, souvent à travers de multiples
    commits, afin de les communiquer clairement aux utilisateurs.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorer les dépréciations

  %p
    Lorsque l'on passe d'une version d'un logiciel à une autre, il devrait être
    très clair si quelque chose peut ne plus fonctionner. Il devrait être
    possible de mettre à jour vers une version qui offre une liste des
    fonctionnalités dépréciées, de retirer ce qui est déprécié, puis de mettre
    à jour vers la version où les dépréciations deviennent des suppressions de
    fonctionnalités.

  %p
    Si vous ne faites rien d'autre, listez les dépréciations, les supressions de
    fonctionnalités, et tous les changements non rétrocompatibles dans votre
    changelog.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Dates confuses

  %p
    Aux États-Unis, les gens mettent le mois en premier (<code>06-02-2012</code>
    pour le 2 juin 2012), tandis que beaucoup de gens dans le reste du monde
    l’écrivent de la façon robotique suivante « 2 Juin 2012 », alors qu’ils le
    prononcent différement. <code>2012-06-02</code> fonctionne logiquement, du plus grand
    au plus petit, ne laisse pas place à la confusion avec un autre format et
    est un #{link_to "standard ISO", data.links.isodate}. C'est pour cela que ce format de date
    est recommandé pour les changelogs.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Inconsistent Changes

  %p
    Un changelog qui ne mentionne qu'une portion des changements peut être aussi 
    dangereux que l'absence totale de changelog. Bien que la majorité des 
    changements peuvent être sans conséquence, par exemple retirer un simple 
    espace dans un texte ne nécessite probablement pas de mention, tout 
    changements important devrait être mentionné dans le changelog. Sans 
    consistence dans l'application des changements, vos utilisateurs pourraient 
    être induit en erreur et penser que le changelog est la seule source de 
    d'information. Ça devrait être le cas. Son pouvoir implique des 
    responsabilités: un bon changelog est synonyme avec un changelog mis à jour 
    de façon consistante.

  %aside
    Il y en a d’autres. Aidez-moi à collecter ces antipatrons en
    #{link_to "ouvrant une issue", data.links.issue} ou une pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Questions fréquemment posées

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Y a-t-il un format de changelog standard ?

  %p
    Pas vraiment. Il y a le guide de style GNU pour les changelogs, ou les
    instructions GNU de deux paragraphes pour les fichiers NEWS. Ces deux
    ressources sont inappropriées et insuffisantes.

  %p
    Ce projet vise à devenir
    = link_to "une meilleure convention pour les changelogs.", data.links.changelog
    Il a pour origine l'observation et le rassemblement des bonne pratiques dans
    la communauté open source.

  %p
    Les critiques constructives, discussions et suggestions pour améliorer ce
    projet #{link_to "sont les bienvenues.", data.links.issue}.


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Comment doit-être nommé le fichier de changelog ?

  %p
    Nommez-le <code>CHANGELOG.md</code>. Certains projets utilisent
    <code>HISTORY</code>, <code>NEWS</code> ou <code>RELEASES</code>.

  %p
    Même s'il est facile d'imaginer que le nom d'un fichier de changelog n'a
    pas grand intérêt, pourquoi rendre la tâche plus difficile que nécessaire
    aux utilisateurs qui cherchent à trouver les changements notables de votre
    projet ?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Quid de GitHub Releases ?

  %p
    C'est une excellente initiative. #{link_to "Releases", data.links.github_releases} peut être
    utilisé pour transformer de simples tags git (par exemple un tag nommé
    <code>v1.0.0</code>) en notes de publication riches soit en ajoutant
    manuellement des notes soit en utilisant les messages de tags gits annotés
    et en les transformant en notes.

  %p
    GitHub Releases crée un changelog non-portable qui n'est visible par les
    utilisateurs qu'à l'intérieur du contexte de GitHub. Il est possible de
    suivre le même format que Keep a Changelog, mais c'est plus difficile.

  %p
    La version actuelle de GitHub Releases est également difficile à découvrir
    pour les utilisateurs, contrairement aux fichiers en majuscules typiques
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.). Un autre souci
    mineur est que l'interface n'offre actuellement pas de lien vers les
    journaux gits entre chaque publication.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Les changelogs peuvent-ils être parsés automatiquement ?

  %p
    C'est difficile, parce que les gens suivent des formats et utilisent des noms
    de fichiers très différents.

  %p
    #{link_to "Vandamme", data.links.vandamme} est une Ruby gem créée par l'équipe
    Gemnasium qui parse les changelogs de
    beaucoup (mais pas tous) de projets open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Qu'en est-il des publications « yanked » (retirées) ?

  %p
    Les publications yanked sont des version qui ont dû être retirées du fait d'un
    sérieux bug ou d'un problème de sécurité. Souvent ces version n'apparaissent
    même pas dans les changelogs. Elles devraient. Voilà comment les afficher :

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Le tag <code>[YANKED]</code> n'est pas mis en avant pour rien. Il est
    important qu'il soit remarqué. Il est également plus facile à digérer pour
    un programme puisqu'il est entouré par des crochets.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Devriez-vous réécrire un changelog ?

  %p
    Bien sûr. Il y a toujours de bonnes raisons d'améliorer un changelog.
    J'ouvre souvent des pull requests pour ajouter des publications manquantes sur
    des projets open source avec des changelogs non-maintenus.

  %p
    Il est aussi possible que vous découvriez que vous aviez oublié de mentionner
    un changement majeur dans les notes de version. Il est évidemment important
    que vous mettiez votre changelog à jour dans ce cas.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Comment puis-je contribuer ?

  %p
    Ce document n'est pas la **vérité**; c'est mon opinion soigneusement
    réfléchie, accompagnée d'informations et d'exemples que j'ai rassemblés.

  %p
    C'est parce que je veux que notre communauté atteigne un consensus. Je crois
    que la discussion est aussi importante que le résultat final.

  %p
    Donc s'il vous plaît, <strong>#{link_to "participez", data.links.repo}</strong>.

.press
  %h3 Conversations
  %p
    J'ai participé au podcast #{link_to "The Changelog", data.links.thechangelog}
    pour expliquer pourquoi les mainteneurs et contributeurs doivent se
    soucier des changelogs et également des motivations derrière ce projet.
```

## File: source/hr/1.0.0/index.html.haml
```haml
---
title: Vodite changelog
description: Ne dajte prijateljima da trpaju git logove u changelogove.
language: hr
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Verzija
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Što je changelog?

  %p
    Changelog je datoteka koja sadrži uređeni kronološki
    poredani popis značajnih promjena unutar svake verzije projekta.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Zašto voditi changelog?

  %p
    Kako bi korisnici i suradnici detaljno vidjeli
    značajne promjene među pojedinim izdanjima (ili verzijama)
    projekta.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kome treba changelog?

  %p
    Ljudima. Bilo da su uobičajeni korisnici ili programeri, krajnji su korisnici
    softvera ljudska bića kojima je stalo do toga od čega je sastavljen. Kada
    se softver mijenja, korisnici žele znati kako i zašto.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Kako kreirati dobar changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Glavna načela

  %ul
    %li
      Changelogs služi <em> ljudima</em>, ne strojevima.
    %li
      Potrebno je stvoriti unos za svaku verziju.
    %li
      Slične promjene potrebno je grupirati.
    %li
      Verzije i sekcije trebaju imati poveznicu.
    %li
      Posljednja verzija treba biti na prvom mjestu.
    %li
      Datum izdavanja svake pojedine verzije treba biti vidljiv.
    %li
      Navesti prati li se #{link_to "Semantičko verzioniranje", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Vrste promjena

  %ul
    %li
      %code Added
      za nove funkcionalnosti.
    %li
      %code Changed
      za promjene u postojećim funkcionalnostima.
    %li
      %code Deprecated
      za funkcionalnosti koje će se ukloniti u budućim verzijama.
    %li
      %code Removed
      za uklonjene funkcionalnosti.
    %li
      %code Fixed
      za ispravke bugova.
    %li
      %code Security
      za slučaj sigurnosnih propusta.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Kako održavati changelog sa što manje napora?

  %p
    Na vrh postavite <code>Unreleased</code> sekciju gdje ćete navoditi nadolazeće
    promjene.

  %p To radimo iz dva razloga:

  %ul
    %li
      Korisnici mogu vidjeti promjene koje mogu očekivati u nadolazećim izdanjima
    %li
      Kod izdavanja nove verzije, moguće je <code>Unreleased</code> sekciju
      samo preimenovati u novo izdanje.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Može li changelog biti loš?

  %p Naravno. Postoji nekoliko slučajeva u kojima changelog može biti beskoristan.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffovi

  %p
    Korištenje commit log diffova u svrhu changeloga nije dobra ideja:
    puni su šuma. Npr. merge commitovi, commitovi s nejasnim naslovima,
    promjene u dokumentaciji i sl.

  %p
    Svrha commita je da bilježi korake u razvoju izvornog koda.
    U nekim projektima se comittovi čiste, no ponekad i ne.

  %p
    Svrha unosa u changelogu je da bilježi značajne razlike, a
    često kroz više commitova i jasno ih prenese krajnjem
    korisniku.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignoriranje uklonjenih funkcionalnosti

  %p
    Kad korisnici nadograde softver na noviju verziju, treba biti potpuno
    jasno da postoji mogućnost da će se neki dio pokvariti. Softver treba biti
    moguće nadograditi na verziju koja će navesti funkcionalnosti koje trebaju
    biti uklonjene, uklanja takve te se kasnije može nadograditi na verziju
    gdje su već uklonjene.

  %p
    U najmanju ruku, potrebno je u changelogu navoditi funkcionalnosti koje će
    biti uklonjene, funkcionalnosti koje su uklonjene i promjene koje će
    utjecati na rad softvera (breaking change).


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Nejasni datumi

  %p
    Regionalni formati datuma variraju diljem svijeta, pa je često
    teško pronaći format datuma koji će odgovarati svima.
    Prednost datuma u formatu <code>2017-07-17</code> je to što je poredan
    od veće prema manjoj jedinici: godina, mjesec, dan. Ovaj je format također
    teško zamijeniti s drugim regionalnim formatima, za razliku od nekih koji,
    primjerice mijenjaju poziciju oznake mjeseca i dana.
    Iz tog razloga, a i zbog toga što je navedeni format #{link_to "ISO standard", data.links.isodate},
    taj se format preporuča za changelog unose.

  %aside
    Ali to nije sve. Pomozite prikupiti primjere loše prakse
    = link_to "otvorivši issue", data.links.issue
    ili pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Česta pitanja

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Postoji li standardni changelog format?

  %p
    Zapravo ne. Postoji #{link_to "GNU changelog stilski priručnik", data.links.gnustyle},
    i #{link_to "GNU NEWS file", data.links.gnunews}
    "priručnik od dva odlomka". Nijedan nije adekvatan ni dovoljan.

  %p
    Cilj je ovoga projekta
    = link_to "kvalitetniji changelog standard.", data.links.changelog
    Nastao je prikupljanjem primjera dobra prakse u
    open source zajednici.

  %p
    Konstruktivna kritika, raprave i prijedlozi za poboljšanje
    = link_to "su dobrodošli.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Kako nazvati changelog datoteku?

  %p
    Dajemo joj naziv <code>CHANGELOG.md</code>. Neki projekti još koriste
    <code>HISTORY</code>, <code>NEWS</code> ili <code>RELEASES</code>.

  %p
    Iako može djelovati da naziv changelog datoteke i nije toliko
    bitan, čemu otežavati korisnicima da dođu do informacije o promjenama?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Što s GitHub Releases?

  %p
    To je ozbiljna inicijativa. #{link_to "Releases", data.links.github_releases} se mogu koristiti
    kako bi git oznake (npr. git oznaka <code>v1.0.0</code>) pretvorili
    u opširnije bilješke o izdanju, upisujući ih ručno ili pak preuzimajući
    anotirane git oznake i pretvarajući ih u unose.

  %p
    GitHub Releases stvara statični changelog vidljiv korisnicima
    unutar GitHub repozitorija. Moguće ih je urediti u format koji
    bi odgovarao Vodite changelog formatu, no često je nešto opširniji.

  %p
    Trenutna GitHub releases verzija i nije baš sasvim vidljiva
    korisnicima, za razliku od uobičajenih datoteka označenih velikim slovima
    (<code>README</code>, <code>CONTRIBUTING</code>, itd.). Još je jedan
    manji problem što trenutno sučelje ne nudi poveznice na commit logove
    između izdanja.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Mogu li changelogovi automatski parsati?

  %p
    Teže, jer se koriste vrlo različiti formati, kao i nazivi datoteka.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Ruby gem koji je kreirao
    Gemnasium tim i može parsati mnoge (ali
    ne sve) changeloge projekata otvorenog koda.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Što s povučenim izdanjima?

  %p
    Povučena ili 'Yanked' izdanja su verzije koje su uklonjene zbog
    ozbiljnijeg buga ili sigurnosnog propusta. Takva se izdanja najčešće
    i ne pojavljuju u changelogu, iako bi trebala. Trebala bi biti navedena
    na sljedeći način:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> oznaka jasno je istaknuta s razlogom. Bitno je
    da ju je lako primijetiti. Buduće da je okružena zagradama, također ju
    je lakše parsati.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Je li potrebno prepravljati changelog?

  %p
    Naravno. Često postoje dobri razlozi da bismo poboljšali changelog. Ja
    često otvaram pull requestove kako bih dodao nedostajuća izdanja projektima
    otvorenog koda, koji ne održavaju changeloge.

  %p
    Moguće je, također da otkrijete, kako ste zaboravili navesti promjenu koja
    bi utjecala na rad (breaking change). U tom je slučaju, očito, vrlo bitno
    ažurirati changelog.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Kako doprinijeti?

  %p
    Ovaj dokument nije <em>Sveto Pismo</em>; ovo je samo pažljivo
    razmotreno mišljenje, uz informacije i primjere koje sam skupio.

  %p
    Razlog je tome to što želim da zajednica postigne konsenzus. Vjerujem,
    također, da je i sama rasprava bitna kao i krajnjni rezultat.

  %p
    Zato, molimo <strong>#{link_to "uskočite", data.links.repo}</strong>.

.press
  %h3 Razgovori
  %p
    Gostovao sam na #{link_to "The Changelog podcastu", data.links.thechangelog}
    gdje sam pokušao objasniti zašto začetnici projekata i suradnici trebaju
    brinuti o changelogovima te motivaciji iza ovog projekta.
```

## File: source/id-ID/1.0.0/index.html.haml
```haml
---
title: Pencatatan Changelog
description: Standarisasi pencatatan Changelog untuk kolaborasi yang lebih baik
language: id-ID
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versi
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Apa itu changelog?

  %p
    Changelog adalah sebuah file yang berisi daftar perubahan yang
    diurutkan secara kronologis untuk setiap versi dari sebuah proyek.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Kenapa perlu untuk mencatat changelog?

  %p
    Untuk mempermudah pengguna dan kontributor melihat perubahan
    apa saja yang terjadi antara setiap rilis (atau versi) dari sebuah proyek.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Siapa yang membutuhkan changelog?

  %p
    Semua orang membutuhkannya. Baik pengguna ataupun pengembang, setiap
    orang yang menggunakan perangkat lunak adalah manusia yang peduli tentang
    apa yang ada di dalam perangkat lunak tersebut. Ketika perangkat lunak berubah,
    mereka ingin tahu apa yang berubah dan mengapa.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Bagaimana cara membuat changelog yang baik?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Prinsip-prinsip Dasar

  %ul
    %li
      Changelog ditulis untuk <em>manusia</em>, bukan mesin.
    %li
      Harus ada catatan untuk setiap versi.
    %li
      Setiap tipe perubahan yang sama harus dikelompokkan.
    %li
      Versi dan seksi harus dapat dirujuk.
    %li
      Versi yang terakhir harus ditulis di paling atas.
    %li
      Tanggal rilis setiap versi harus ditulis.
    %li
      Berikan informasi jika kalian menggunakan #{link_to "Semantic Versioning", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Jenis-jenis perubahan

  %ul
    %li
      %code Added/Ditambahkan
      untuk fitur yang baru.
    %li
      %code Changed/Diubah
      untuk perubahan di fitur yang sudah ada.
    %li
      %code Deprecated/Akan Dihilangkan
      untuk fitur yang akan dihapus dalam waktu dekat.
    %li
      %code Removed/Dihilangkan
      untuk fitur yang sudah dihapus.
    %li
      %code Fixed/Diperbaiki
      untuk setiap perbaikan bugs.
    %li
      %code Security/Keamanan
      jika ada celah keamanan.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Apa yang bisa saya lakukan untuk mempermudah pemeliharaan changelog?

  %p
    Sisakan bagian <code>Unreleased/Belum Dirilis</code> di bagian paling atas
    file changelog untuk mencatat perubahan yang akan datang.


  %p Hal ini berguna untuk dua hal:

  %ul
    %li
      Orang-orang bisa melihat perubahan apa saja yang akan datang.
    %li
      Saat waktu rilis datang, tinggal pindahkan bagian <code>Unreleased/Belum Dirilis</code>
      ke catatan rilis versi baru di bawah.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Apakah changelog bisa menjadi tidak bermanfaat?

  %p Bisa, berikut beberapa skenario ketika changelog menjadi tidak bermanfaat:

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Menggunakan Commit log diffs sebagai changelog

  %p
    Menggunakan commit log diffs (catatan perbedaan setiap commit) bisa
    membuat changelog susah untuk dibaca. Commit dengan judul yang tidak jelas,
    dokumentasi perubahan, dan sebagainya, malah akan membuat
    changelog terlalu berisik dan susah dibaca.

  %p
    Tujuan utama dari commit adalah untuk mencatat setiap perubahan dari source
    code. Beberapa proyek merapikan commitnya, beberapa tidak.

  %p
    Tujuan dari changelog adalah untuk mencatat perubahan yang pantas
    untuk dicatat, bisa jadi beberapa commit dijadikan satu catatan
    untuk lebih memudahkan pembaca.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Mengabaikan Deprecations (fitur yang akan dihilangkan)

  %p
    Saat menaikkan versi, harus ditulis dengan jelas apa saja
    yang kira-kira bisa membuat sistem tidak berjalan. Sebaiknya
    terdapat versi yang mencatat apa saja yang akan dihilangkan,
    lalu menghapus fitur yang dihilangkan, dan naikkan lagi ke versi
    dengan fitur yang sudah dihilangkan.

  %p
    Jika kalian tidak mengubah apapun, tetap catat fitur yang akan
    dihilangkan, fitur yang sudah dihilangkan, dan perubahan-perubahan
    lain yang bisa membuat sistem tidak berjalan.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Perbedaan Format Tanggal

  %p
    Format tanggal regional berbeda-beda sesuai dengan budaya masing-masing,
    dan seringkali perbedaan ini susah untuk dipahami dan dimengerti.
    Penggunaan format tanggal <code>2017-07-17</code> lebih mudah untuk dimengerti,
    karena diurutkan berdasarkan unit terbesar: tahun, bulan, dan tanggal.
    Format ini juga merupakan #{link_to "Standar ISO", data.links.isodate}, sehingga
    inilah yang dipakai untuk pencatatan changelog.

  %aside
    Ada beberapa permasalahan lainnya, bantu kami dengan beberapa antipatterns
    dengan #{link_to "mengirimkan issue", data.links.issue} atau kirimkan pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Pertanyaan yang Sering Ditanyakan

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Apakah ada standar untuk format changelog?

  %p
    Tidak, ada format GNU untuk changelog, atau format 2 paragraf GNU NEWS.
    Keduanya tidak benar-benar cukup, gunakan format yang disetujui tim masing-masing.

  %p
    Proyek ini ditujukan untuk
    = link_to "membuat aturan changelog yang lebih baik", data.links.changelog
    berdasarkan observasi beberapa changelog di komunitas open source
    dan menyatukan mereka.

  %p
    Kritik yang membangun, diskusi, dan saran untuk perbaikan
    = link_to "sangat diterima.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Apa nama file yang cocok untuk file changelog?

  %p
    Gunakan nama <code>CHANGELOG.md</code>, beberapa proyek
    menggunakan <code>HISTORY</code>, <code>NEWS</code> atau <code>RELEASE</code>.

  %p
    Sebenarnya tidak terlalu susah untuk menamai file changelog, cukup
    berikan nama yang mudah dikenali oleh orang-orang
    supaya mudah untuk dibaca.

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Apa itu GitHub Releases?

  %p
    #{link_to "Github Release", data.links.github_releases} adalah salah satu
    insiatif dari GitHub untuk membuat changelog berdasarkan git tags,
    contohnya, tag dengan nama <code>v1.0.0</code>. Isi dari changelog
    bisa ditulis manual atau menggunakan pesan yang ditulis bersamaan dengan
    tags.

  %p
    GitHub Releases membuat changelog yang tidak portable dan hanya bisa
    bekerja dengan baik di lingkup GitHub. Sangat mungkin untuk
    membuat GitHub Releases terlihat mirip dengan format pencatatan changelog
    yang dijelaskan di sini, tapi butuh usaha ekstra.

  %p
    Versi GitHub releases yang sekarang juga tidak terlalu umum
    untuk orang-orang, dan hanya bisa dibuka melalui sub menu
    di GitHub, berbeda dengan file-file seperti
    (<code>README</code>, <code>CONTRIBUTING</code>, dsb.) yang
    langsung terlihat saat proyek pertama kali dibuka. Keterbatasan lainnya
    adalah tidak adanya link di GitHub releases.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Apakah Changelog Bisa Diparse Secara Otomatis?

  %p
    Susah, karena orang-orang menggunakan versi dan format
    changelog yang berbeda-beda.

  %p
    #{link_to "Vandamme", data.links.vandamme} adalah Ruby Gem yang dibuat oleh tim
    Gemnasium yang bisa mem-parsing (tetapi tidak semua) changelog proyek open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Bagaimana dengan Rilis YANKED (rilis yang dibatalkan)?

  %p
    Rilis yang dibatalkan adalah rilis yang dibatalkan, bisa jadi
    karena ada bug yang fatal dan permasalahan keamanan. Versi
    ini sering tidak dimasukkan ke changelog, padahal seharusnya ditulis
    sebagaimana berikut.

  %p <code>## 0.0.5 - 2014-12-13 [YANKED/DIBATALKAN]</code>

  %p
    Tag <code>[YANKED/DIBATALKAN]</code> ditulis dengan jelas supaya
    orang memperhatikannya, dan dikurung dengan kurung kotak supaya
    mudah untuk diparse.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Bolehkah Menulis Ulang Changelog?

  %p
    Tentu saja, ada banyak alasan bagus untuk menulis ulang changelog,
    salah satunya untuk rilis-rilis yang lupa untuk dituliskan di
    beberapa proyek.

  %p
    Juga sangat mungkin saat menulis ulang, kalian ingat tentang perubahan
    yang bisa membuat sistem tidak bekerja yang belum dituliskan. Dalam hal ini,
    sangat penting untuk mengubah changelog supaya datanya lebih akurat.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Bagaimana Saya Bisa Berkontribusi?

  %p
    Dokumen ini bukan kebenaran absolut, ini hanyalah opini yang dikumpulkan
    dari beberapa informasi dan contoh yang kami kumpulkan.

  %p
    Ini karena kami ingin komunitas terus berdiskusi untuk mencapai konsensus
    yang terbaik untuk pencatatan changelog.

  %p
    Jadi silahkan, <strong>#{link_to "kirimkan saran", data.links.repo}</strong>.

.press
  %h3 Percakapan
  %p
    Kami mengisi acara di #{link_to "Podcast Changelog", data.links.thechangelog}
    untuk berbicara tentang kenapa maintainer dan kontributor harus peduli
    dengan changelog, dan motivasi dari proyek ini.
```

## File: source/id/1.1.0/index.html.haml
```haml
---
title: Pencatatan Changelog
description: Standarisasi pencatatan Changelog untuk kolaborasi yang lebih baik
language: id
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Apa itu changelog?

  %p
    Changelog adalah sebuah file berisi perubahan penting setiap versi sebuah proyek yang kronologis dan terkurasi. 

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Kenapa mencatat changelog?

  %p
    Agar perubahan penting antar versi sebuah proyek lebih mudah diamati bagi pengguna maupun kontributor.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Siapa yang membutuhkan changelog?

  %p
    Kita semua. Baik itu pengguna atau pengembang, konsumen perangkat lunak adalah manusia yang peduli akan isi perangkat lunak tersebut. Kita semua ingin mengetahui bagaimana perangkat lunak tersebut berubah dan alasan dibalik perubahan tersebut. 

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Bagaimana cara mencatat changelog yang baik?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Panduan Dasar

  %ul
    %li
      Changelog dicatat <em>untuk manusia</em>, bukan mesin.
    %li
      Ada catatan untuk setiap versi.
    %li
      Perubahan dengan jenis yang sama dikelompokkan.
    %li
      Tercantum rujukan untuk versi dan seksi.
    %li
      Versi terbaru tercatat di bagian teratas.
    %li
      Tanggal rilis setiap versi tercatat.
    %li
      Sebutkan apakah anda mengikuti #{link_to "Semantic Versioning", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Jenis Perubahan

  %ul
    %li
      %code Added/Ditambahkan
      untuk fitur baru.
    %li
      %code Changed/Diubah
      untuk perubahan pada fitur yang sudah ada.
    %li
      %code Deprecated/Akan Dihilangkan
      untuk fitur yang akan dihilangkan dalam waktu dekat.
    %li
      %code Removed/Dihilangkan
      untuk fitur yang telah dihilangkan.
    %li
      %code Fixed/Diperbaiki
      untuk perbaikan bugs.
    %li
      %code Security/Keamanan
      jika ada kerentanan.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Bagaimana cara mempermudah pemeliharaan changelog?

  %p
    Sediakan bagian <code>Unreleased/Belum Dirilis</code> di atas file changelog untuk mencatat perubahan yang akan datang.

  %p Manfaat bagian ini:

  %ul
    %li
      Kita dapat melihat perubahan yang akan datang.
    %li
      Bagian <code>Unreleased/Belum Dirilis</code> dapat dipindahkan ke catatan versi terbaru saat sudah rilis.
.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Apakah bisa changelog menjadi tidak bermanfaat?

  %p Bisa. Berikut sedikit contoh bagaimana keberadaan changelog menjadi tidak membantu:

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Menggunakan pesan commit log diff seadanya

  %p
    Pesan commit log diff (perbedaan sebuah file antar versi) merupakan catatan yang buruk untuk dijadikan changelog karena seringkali mereka tidak bermakna. Contohnya seperti merge commit, commit berjudul samar, perubahan dokumentasi, dan lainnya.

  %p
    Tujuan pesan commit adalah mendokumentasikan sebuah tindakan dalam source code. Pesan ini bisa jadi tidak dirapikan.
  %p
    Tujuan sebuah catatan dalam changelog adalah mendokumentasikan perubahan-perubahan yang patut diperhatikan, dimana catatan ini seringkali merangkum perubahan serangkaian commit agar mudah dimengerti. 

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Mengabaikan Deprecations (fitur yang akan dihilangkan)

  %p
    Jika ada perubahan yang tidak kompatibel dari satu versi ke versi lainnya, perubahan ini seharusnya teramat jelas bagi orang-orang. Saat kita menggunakan sebuah versi perangkat lunak yang di dalamnya terdapat fitur yang akan dihilangkan (Deprecated), seharusnya kita dapat menghapus fitur tersebut lalu melakukan upgrade ke versi dimana fitur tersebut sudah dihilangkan (Removed). 

  %p
    Catat fitur yang akan dihapus, telah dihapus, dan perubahan tidak kompatibel pada changelog anda jika anda tidak mencoba hal tersebut.  


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Perbedaan Format Tanggal

  %p
    Mencari pola tanggal yang intuitif dan mudah dipahami oleh semua orang adalah masalah yang sulit. Hal ini karena pola tanggal di satu belahan bumi akan berbeda dengan belahan bumi lainnya. Tanggal yang disusun dalam pola <code>2017-07-17</code> memiliki keuntungan dimana pola tersebut dimulai dari unit terbesar: tahun, bulan, lalu hari. Format ini juga tidak tumpang tindih dengan format tanggal lainnya. Berbeda dengan format tanggal regional yang terkadang menukar posisi angka bulan dan tanggal.
    Format tanggal ini direkomendasikan dalam catatan changelog bukan hanya karena alasan yang telah disebut, tetapi juga karena pola ini mengikuti #{link_to "ISO standard", data.links.isodate} untuk penulisan tanggal.  

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Pencatatan yang tidak konsisten

  %p
    Catatan changelog yang tidak lengkap sama bahayanya dengan tidak mencatatnya sama sekali. Walau perubahan kecil - menghapus sebuah spasi misalnya - mungkin tidak perlu dicatat, semua perubahan penting yang patut diperhatikan harus disebut dan tercatat di changelog. Jika dilakukan setengah-setengah konsumen perangkat lunak anda akan berpikir mereka sedang membaca changelog yang benar, padahal kenyataannya salah. Changelog seharusnya selalu benar. Changelog yang baik adalah changelog yang selalu diperbarui. Jika anda ingin memiliki changelog yang baik, inilah tanggung jawab anda.

  %aside
    Masih banyak contoh lainnya. Bantu saya mengumpulkannya dengan 
    = link_to "membuka sebuah issue", data.links.issue
    atau pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Pertanyaan yang Sering Diajukan

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Apakah ada format standar untuk sebuah changelog?

  %p
    Tidak, sih. Ada #{link_to "Panduan Ragam Penulisan Changelog GNU", data.links.gnustyle},
    atau "panduan" #{link_to "dua paragraf untuk GNU NEWS file", data.links.gnunews}. Tapi keduanya antara tidak mumpuni atau tidak lengkap.

  %p
    Proyek ini ingin menghasilkan
    = link_to "konvensi changelog yang lebih baik.", data.links.changelog
    Hal-hal yang diutarakan merupakan sekumpulan pengamatan berbagai disiplin baik yang telah diadopsi oleh komunitas open source lain. 
    
  %p
    Kritik sehat, saran, dan diskusi untuk meningkatkan proyek ini  
    = link_to "dipersilakan.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Sebaiknya file changelog diberi nama apa?

  %p
    Sebut saja <code>CHANGELOG.md</code>. Beberapa proyek menyebutnya
    <code>HISTORY</code>, <code>NEWS</code> or <code>RELEASES</code>.

  %p
    Walau perihal nama ini sepertinya sepele, kenapa pengguna perangkat lunak anda yang mencari perubahan penting harus dibuat sulit?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Bagaimana dengan GitHub Releases?

  %p
    Github Releases merupakan upaya yang baik. #{link_to "Releases", data.links.github_releases} dapat merubah git tag (contohnya sebuah tag dengan judul <code>v1.0.0</code>)
    menjadi catatan rilis yang informatif dengan menambahkan catatan secara manual atau dengan memanfaatkan pesan-pesan git tag dan merubahnya menjadi catatan.

  %p
    Changelog yang dihasilkan oleh GitHub Releases bermanfaat hanya dalam konteks pengguna GitHub. Hasil catatannya dapat dibuat menyerupai format Pencatatan Changelog namun akan rumit.

  %p
    GitHub releases yang sekarang tidak begitu mudah ditemukan oleh pengguna, berbeda dengan file-file (<code>README</code>, <code>CONTRIBUTING</code>, dll.). Tautan menuju commit log antar versi rilis juga tidak ditampilkan di antarmuka pengguna. 

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Apakah changelog dapat diuraikan secara otomatis?

  %p
    Sulit, karena orang-orang mengikuti panduan format dan nama file yang berbeda-beda.

  %p
    #{link_to "Vandamme", data.links.vandamme} adalah sebuah Ruby gem yang diciptakan oleh tim Gemnasium dan mampu menguraikan changelog proyek-proyek open source (tapi tidak semua).


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Bagaimana dengan rilis yang dibatalkan (YANKED)?

  %p
    Rilis bertanda Yanked artinya versi tersebut ditarik karena adanya bug atau masalah kerentanan yang serius. Versi-versi ini harusnya dicatat. Tetapi seringkali tidak. Berikut cara mencatatnya:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Tag <code>[YANKED]</code> ditulis sedemikian rupa. Hal ini karena orang-orang harus melihatnya. Selain huruf kapital, dengan menulisnya di dalam kurung siku akan mempermudah penguraian catatan ini secara programatis.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Bolehkah anda menulis ulang sebuah changelog?
  %p
    Boleh saja. Memperbarui sebuah changelog merupakan ide yang baik. Saya rutin membuat pull request untuk menambahkan versi-versi yang tidak tercatat dalam changelog proyek open source yang tidak dirapikan.

  %p
    Memperbarui dan mencermati changelog itu penting. Apalagi jika anda lupa untuk mencatat perubahan yang tidak kompatibel pada versi tertentu. 


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Bagaimana saya dapat berkontribusi?

  %p
    Dokumen ini bukanlah sesuatu yang <strong>mutlak</strong>; melainkan sebuah opini yang terbentuk setelah saya mempertimbangkan banyak hal, meliputi informasi dan contoh-contoh yang saya kumpulkan.

  %p
    Saya ingin komunitas ini mencapai sebuah mufakat. Saya yakin diskusi yang muncul untuk menghasilkan sebuah keputusan adalah sama pentingnya dengan keputusan tersebut.

  %p
    Jadi, silahkan kirim <strong>#{link_to "saran", data.links.repo}</strong>.

.press
  %h3 Percakapan
  %p
    Saya berbicara tentang mengapa kontributor dan maintainer seharusnya mencermati changelog di #{link_to "The Changelog podcast", data.links.thechangelog}, disana saya juga berbicara tentang motivasi di belakang proyek ini.
```

## File: source/it-IT/0.3.0/index.html.haml
```haml
---
title: Mantenere un CHANGELOG
description: Non lasciate che i vostri amici copino e incollino i git log nei CHANGELOG
language: it-IT
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Cos'è un change log?
  Un change log è un file che contiene una lista curata e ordinata cronologicamente
  delle modifiche degne di nota per ogni versione di un progetto.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### A cosa serve un change log?
  Per rendere facile agli utilizzatori e contribuenti vedere con precisione quali modifiche
  importanti sono state fatte tra ogni release (o versione) del progetto.

  ### Perché dovrebbe importarmi?
  Perché gli strumenti software sono fatti per le persone.
  Se non ti importa, perché contribuisci all'open source?
  Di certo ci deve essere un "kernel" (ha!) di interesse
  da qualche parte in quel tuo amorevole cervello.

  [Nel podcast "The Changelog" con Adam Stacoviak e Jerod Santo][thechangelog]
  (titolo appropriato, vero?) ho parlato del perché maintainer e contribuenti
  dovrebbero esserne interessati, e delle motivazioni dietro a questo progetto.
  Se avete tempo (1:06), vale la pena ascoltarlo.

  ### Come si può fare un buon change log?
  Sono contento che tu l'abbia chiesto.

  Un buon change log è guidato dai seguenti principi:

  - È fatto per umani, non macchine, quindi la leggibilità è cruciale.
  - Facile da linkare ad altre sezioni (da cui il Markdown invece che testo normale).
  - Una sotto-sezione per ogni versione.
  - Elenca le release in ordine cronologico inverso (quelle più recenti all'inizio).
  - Scrive tutte le date nel formato `YYYY-MM-DD`. (Esempio: `2012-06-02` sta per `2 Giugno 2012`). È internazionale, [sensato](https://xkcd.com/1179/), e indipendente dalla lingua.
  - Dichiara esplicitamente se il progetto segue il [Semantic Versioning][semver].
  - Ogni versione dovrebbe:
    - Elencare la sua data di rilascio nel formato sopra specificato.
    - Raggruppare le modifiche per descrivere il loro impatto sul progetto, come segue:
      - `Added` per le nuove funzionalità.
      - `Changed` per le modifiche a funzionalità esistenti.
      - `Deprecated` per vecchie feature stabili che verranno rimosse nelle future release.
      - `Removed` per funzionalità precedentemente deprecate rimosse in questa release.
      - `Fixed` per tutti i bug fix.
      - `Security` per invitare gli utilizzatori ad aggiornare in caso di vulnerabilità.

  ### Come posso minimizzare lo sforzo richiesto?
  Usa sempre una sezione `"Unreleased"` all'inizio per tenere traccia di qualsiasi modifica.

  Questo serve per due scopi:

  - La gente può vedere quali modifiche si può aspettare in future release
  - Ad una nuova release, si deve solo sostituire `"Unreleased"` con il numero di versione
    e aggiungere una nuova sezione `"Unreleased"` all'inizio.

  ### Cosa fa piangere gli unicorni?
  Bene...vediamo un po'.

  - **Copia&incolla un diff di commig log.** Non fatelo, così non aiutate nessuno.
  - **Non enfatizzare le funzioni deprecate.** Quando le persone aggiornano da una versione ad un'altra,
    dovrebbe essere dolorosamente chiaro quando qualcosa si romperà.
  - **Date in formati specifici per regione.** Negli Stati Uniti si mette prima il mese nelle date
    ("06-02-2012" sta per 2 Giugno 2012, che non ha senso), mentre molte persone
    nel resto del mondo scrivono un robotico "2 June 2012", ma lo pronunciano diversamente.
    "2012-06-02" funziona con la logica dal più grande al più piccolo, non ha sovrapposizioni
    ambigue con altri formati di date, ed è uno [standard ISO](http://www.iso.org/iso/home/standards/iso8601.htm).
    Per tutti questi motivi, è il formato di date raccomandato per i change log.

  C'è di più. Aiutatemi a collezionare queste "lacrime di unicorno" [aprendo una "issue"][issues]
  o una pull request.

  ### Esiste un formato standard per i change log?
  Purtroppo no. Calma. So che state furiosamente correndo a cercare quel link
  alla guida di stile per i change log GNU, o alle linee guida per or il file a due paragrafi GNU NEWS.
  La guida GNU è un buon punto di partenza, ma è tristemente ingenuo.
  Non c'è nulla di male nell'essere ingenuo, ma quando le persone cercano una guida, questa caratteristica
  è raramente d'aiuto. Specialmente se ci sono molte situazioni e casi limite da gestire.

  Questo progetto [contiene ciò che spero diventerà una migliore convenzione per i file CHANGELOG][CHANGELOG].
  Non credo che lo status quo sia sufficientemente buono, e penso che noi, come comunità,
  possiamo arrivare a convenzioni migliori se tentiamo di estrarre le pratiche migliori
  da progetti software reali. Vi consiglio di guardarvi intorno e ricordare che
  [ogni discussione e suggerimento per migliorare sono sempre benvenuti][issues]!

  ### Come si dovrebbe chiamare il file contenente il change log?
  Se non l'avete dedotto dall'esempio qui sopra, `CHANGELOG.md` è la convenzione migliore finora.

  Alcuni progetti usano anche `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`, etc.

  È un disastro. Tutti questi nomi contribuiscono solo a rendere più difficile trovarlo.

  ### Perché la gente non si limita ad usare diff di `git log`?
  Perché i log diff sono pieni di rumore, per loro natura. Non potrebbero creare un change log
  decente nemmeno in un ipotetico progetto gestito da perfetti umani che non fanno mai errori
  di digitazione, non dimenticano mai di fare commit dei nuovi file, non dimenticano mai
  alcuna parte di un refactoring.
  Lo scopo di un commit è di documentare un passo atomico nel processo di evoluzione del codice
  da uno stato ad un altro. Lo scopo di un change log è di documentare le differenze
  degne di nota tra questi stati.

  La differenza tra un change log e un commit log è
  come la differenza tra un buon commento nel codice e il codice stesso:
  uno descrive il *perché*, l'altro il come.

  ### Si possono analizzare automaticamente i change log?
  È difficile, perché le persone usano formati e nomi di file profondamente diversi.

  [Vandamme][vandamme] è una Ruby gem
  creata dal team [Gemnasium][gemnasium] ed è in grado di fare il parsing dei
  change log di molti (ma non tutti) i progetti open source.

  ### Perché si alterna tra i nomi "CHANGELOG" e "change log"?
  "CHANGELOG" è il nome del file. È un po' invadente ma è una
  convenzione storica seguita da molti progetti open source.
  Altri esempi di file simili includono [`README`][README], [`LICENSE`][LICENSE],
  e [`CONTRIBUTING`][CONTRIBUTING].

  I nomi in maiuscolo (che su vecchi sistemi operativi erano ordinati per primi)
  vengono usati per attirare l'attenzione su di essi. Poiché sono metadati
  importanti relativi al progetto, possono essere utili per chiunque voglia usarlo
  o contribuire ad esso, un po' come i [badge dei progetti open source][shields].

  Quando mi riferisco a un "change log", sto parlando della funzione di questo file:
  registrare le modifiche.

  ### Cosa sono le release "yanked"?
  Le release "yanked" sono versioni che sono state rimosse a causa di bug seri
  o problemi di sicurezza. Spesso queste versioni non appaiono nei change
  log. Invece dovrebbero esserci, nel seguente modo:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  L'etichetta `[YANKED]` è in maiuscolo per un motivo. È importante che la gente la noti.
  Poiché è racchiusa tra parentesi quadre è anche più semplice da riconoscere nel parsing automatizzato.

  ### Si dovrebbe mai modificare un change log?
  Certo. Ci sono sempre buoni motivi per migliorare un change log. Io apro regolarmente
  delle pull request per aggiungere release mancanti ai progetti open source che non mantengono
  correttamente i change log.

  È anche possibile che si scopra di aver dimenticato di annotare una modifica
  non retro-compatibile nelle note per una versione. Ovviamente è importante aggiornare
  il change log in questo caso.

  ### Come posso contribuire?
  Questo documento non è la **verità assoluta**; è solo la mia attenta
  opinione, arricchita dalle informazioni ed esempi che ho raccolto.
  Anche se fornisco un [CHANGELOG][] reale sul [repository GitHub][gh],
  ho volutamente evitato di creare una *release* o una chiara lista di regole
  da seguire (come fa, ad esempio, [SemVer.org][semver]).

  Questo è perché voglio che la nostra comunità raggiunga un consenso. Credo che
  la discussione sia importante almeno quanto il risultato finale.

  Quindi per favore [**partecipate**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/it-IT/1.0.0/index.html.haml
```haml
---
title: Tenere un Changelog
description: Non lasciare che i tuoi amici facciano copia incolla dei git logs nei changelog.
language: it-IT
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versione
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Cos'è un changelog?

  %p
    Un changelog è un file che contiene una lista curata e ordinata cronologicamente
    delle modifiche degne di nota per ogni versione di un progetto.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Perché tenere un changelog?

  %p
    Per rendere facile agli utilizzatori e contribuenti vedere con precisione quali modifiche
    importanti sono state fatte tra ogni release (o versione) del progetto.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Chi ha bisogno di un changelog?

  %p
    Le persone ne hanno bisogno. Che si tratti di consumatori o di sviluppatori, gli utenti finali
    sono persone interessate a ciò che avviene in esso.
    Se il software è cambiato, allora le persone vogliono sapere come e perché.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Come posso fare un buon changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Linee guida

  %ul
    %li
      I changelog sono <em>per le persone</em>, non per le macchine.
    %li
      Dovrebbe esserci una voce per ogni singola versione.
    %li
      Le modifiche dello stesso tipo dovrebbero essere raggruppate.
    %li
      Versioni e sezioni dovrebbero essere collegabili.
    %li
      L'ultima versione viene per prima.
    %li
      Viene mostrata la data di release di ogni versione.
    %li
      Si dichiara se il progetto segue il #{link_to "Versionamento Semantico", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipologie di cambiamenti

  %ul
    %li
      %code Added
      per le nuove funzionalità.
    %li
      %code Changed
      per le modifiche a funzionalità esistenti.
    %li
      %code Deprecated
      per le funzionalità che saranno rimosse nelle future release.
    %li
      %code Removed
      per funzionalità rimosse in questa release.
    %li
      %code Fixed
      per tutti i bug fix.
    %li
      %code Security
      per invitare gli utilizzatori ad aggiornare in caso di vulnerabilità.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Come posso ridurre gli sforzi necessari a mantenere un changelog?

  %p
    Tieni una sezione <code>Unreleased</code> in cima al changelog in
    modo da tenere traccia dei cambiamenti imminenti.

  %p Questo serve per due scopi:

  %ul
    %li
      Le persone possono vedere quali modifiche si possono aspettare nelle future release.
    %li
      Ad una nuova release, si deve solo spostare i cambiamenti della sezione
      `"Unreleased"` in una nuova sezione con il corrispettivo numero di versione.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    I changelog possono essere cattivi?

  %p Si. Ecco alcuni modi in cui possono essere meno utili.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    Usare i commit log diffs al posto dei changelog è una brutta idea: contengono solo cose superflue.
    Cose come i merge commits, i commits con titoli oscuri,
    le modifiche della documentazione, etc.

  %p
    Lo scopo di un commit è quello di documentare un passo nell'evoluzione
    del codice sorgente. Alcuni progetti ripuliscono i commit, altri non lo fanno.

  %p
    Lo scopo di una voce changelog è quello di documentare le differenze rilevanti,
    spesso attraverso commit multipli, per comunicarli in modo chiaro agli utenti finali.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorare le deprecazioni
  %p
    Quando le persone aggiornano da una versione ad un'altra,
    dovrebbe essere dolorosamente chiaro che qualcosa si romperà.
    Dovrebbe essere possibile eseguire l'aggiornamento a una versione
    che elenca le deprecazioni, rimuovere ciò che è deprecato, quindi
    aggiornare alla versione in cui le deprecazioni diventano rimozioni.
  %p
    Se non fai nient'altro elenca le deprecazioni, le rimozioni e
    qualsiasi altro cambiamento nel tuo changelog.
  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Confusione delle date

  %p
    I formati di date variano da Paese a Paese e spesso
    trovare un formato human-friendly che sia intuitivo per tutti
    è cosa ardua. Il vantaggio delle date formattate come
    <code>2017-07-17</code> è che seguono l'ordine dal maggiore
    al minore: anno, mese e giorno. Inoltre questo formato non
    ha sovrapposizioni ambigue con altri formati di date, a differenza
    di alcuni formati regionali che scambiano la posizione del mese e del giorno.
    Queste motivazioni e il fatto che questo formato di data è uno
    #{link_to "standard ISO", data.links.isodate} spiegano perché questo è il formato di date
    raccomandato per i changelog.

  %aside
    C'è di più. Aiutatemi a collezionare altre situazioni cattive
    = link_to "aprendo un issue", data.links.issue
    o una pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Domande frequenti
  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Esiste un formato standard per i changelog?

  %p
    Non esattamente. Esistono le linee guida GNU sullo stile dei changelog, oppure
    i due lunghi paragrafi nel documento di GNU NEWS denominato "guideline". Entrambe
    sono inadeguate o insufficienti.

  %p
    Questo progetto vuole essere
    = link_to "una migliore convenzione  per i file changelog.", data.links.changelog
    Per questo motivo osserviamo le migliori pratiche della comunità open source
    e le portiamo avanti.

  %p
    Critiche, discussioni e suggerimenti per migliorare
    = link_to "sono ben accetti.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Come si dovrebbe chiamare il file di changelog?

  %p
    Chiamalo <code>CHANGELOG.md</code>. Alcuni progetti usano anche
    <code>HISTORY</code>, <code>NEWS</code> o <code>RELEASES</code>.

  %p
    Risulta facile pensare che il nome del tuo file changelog
    non sia poi così importante, allora perché non rendere facile la
    vita ai tuoi utenti, usando sempre lo stesso nome?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Cosa dire delle GitHub Releases?

  %p
    È una bella iniziativa. #{link_to "Releases", data.links.github_releases} può essere usato
    per rendere semplice i git tags (per esempio il nome del tag
    <code>v1.0.0</code>) All'interno di note di rilascio ben dettagliate
    si possono aggiungere le note manualmente oppure è possibile
    utilizzare i messaggi dei git tag inserendoli dentro le
    note di rilascio.

  %p
    GitHub Releases crea un changelog non-portable che può essere solo
    visualizzato dagli utenti nel contesto di GitHub. È possibile farlo
    apparire molto simile al formato di Keep a Changelog, tuttavia tende
    ad essere un po' più complicato.

  %p
    La versione corrente di GitHub Releases risulta inoltre
    probabilmente poco rilevante per gli utenti finali, a differenza dei
    tipici file maiuscoli (<code>README</code>,
    <code>CONTRIBUTING</code>, etc.). Un altro problema minore è che
    l'interfaccia non offre attualmente link per la creazione di log tra
    ciascuna release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    I changelog possono essere analizzati automaticamente?

  %p
    È difficile, perché le persone usano formati e nomi di file
    profondamente diversi.

  %p
    #{link_to "Vandamme", data.links.vandamme} è una Ruby gem creata dal team
    Gemnasium ed è in grado di fare il parsing dei
    changelog di molti (ma non tutti) i progetti open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Cosa sono le release "yanked"?

  %p
    Le release "yanked" sono versioni che sono state rimosse a causa di
    bug seri o problemi di sicurezza. Spesso queste versioni non
    appaiono nei changelog. Invece dovrebbero esserci, nel seguente
    modo:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    L'etichetta <code>[YANKED]</code> è in maiuscolo per un motivo.
    È importante che le persone la notino.
    Poiché è racchiusa tra parentesi quadre è anche
    più semplice da riconoscere nel parsing automatico.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Si dovrebbe mai modificare un changelog?

  %p
    Certo. Ci sono sempre buoni motivi per migliorare un changelog. Io apro regolarmente
    delle pull request per aggiungere release mancanti ai progetti open source che non mantengono
    correttamente i changelog.

  %p
    È anche possibile che si scopra di aver dimenticato di annotare una modifica
    non retro-compatibile nelle note per una versione. Ovviamente è importante aggiornare
    il changelog in questo caso.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Come posso contribuire?

  %p
    Questo documento non è la <strong>verità assoluta</strong>; è solo la mia attenta
    opinione, arricchita dalle informazioni ed esempi che ho raccolto.

  %p
    Questo perché voglio che la nostra comunità raggiunga un consenso. Credo che
    la discussione sia importante almeno quanto il risultato finale.

  %p
    Quindi per favore <strong>#{link_to "partecipate", data.links.repo}</strong>.

.press
  %h3 Dialogo
  %p
    Sono andato a #{link_to "The Changelog podcast", data.links.thechangelog}
    per parlare del perché i gestori e i contributori dovrebbero preoccuparsi dei changelog
    e anche delle motivazioni dietro questo progetto.
```

## File: source/it-IT/1.1.0/index.html.haml
```haml
---
title: Tenere un Changelog
description: Non lasciare che i tuoi amici facciano copia incolla dei git logs nei changelog.
language: it-IT
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versione
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Cos'è un changelog?

  %p
    Un changelog è un file che contiene una lista curata e ordinata cronologicamente
    delle modifiche degne di nota per ogni versione di un progetto.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Perché tenere un changelog?

  %p
    Per rendere facile agli utilizzatori e contribuenti vedere con precisione quali modifiche
    importanti sono state fatte tra ogni release (o versione)
    del progetto.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Chi ha bisogno di un changelog?

  %p
    Le persone ne hanno bisogno. Che si tratti di consumatori o di sviluppatori, gli utenti finali
    sono persone interessate a ciò che avviene in esso.
    Se il software è cambiato, allora le persone vogliono sapere come e perché.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Come posso fare un buon changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Linee guida

  %ul
    %li
      I changelog sono <em>per le persone</em>, non per le macchine.
    %li
      Dovrebbe esserci una voce per ogni singola versione.
    %li
      Le modifiche dello stesso tipo dovrebbero essere raggruppate.
    %li
      Versioni e sezioni dovrebbero essere collegabili.
    %li
      L'ultima versione viene per prima.
    %li
      Viene mostrata la data di release di ogni versione.
    %li
      Si dichiara se il progetto segue il #{link_to "Versionamento Semantico", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipologie di cambiamenti

  %ul
    %li
      %code Added
      per le nuove funzionalità.
    %li
      %code Changed
      per le modifiche a funzionalità esistenti.
    %li
      %code Deprecated
      per le funzionalità che saranno rimosse nelle future release.
    %li
      %code Removed
      per funzionalità rimosse in questa release.
    %li
      %code Fixed
      per tutti i bug fix.
    %li
      %code Security
      per invitare gli utilizzatori ad aggiornare in caso di vulnerabilità.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Come posso ridurre gli sforzi necessari a mantenere un changelog?

  %p
    Tieni una sezione <code>Unreleased</code> in cima al changelog in
    modo da tenere traccia dei cambiamenti imminenti.

  %p Questo serve per due scopi:

  %ul
    %li
      Le persone possono vedere quali modifiche si possono aspettare nelle future release.
    %li
      Ad una nuova release, si deve solo spostare i cambiamenti della sezione
      <code>Unreleased</code> in una nuova sezione con il corrispettivo numero di versione.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    I changelog possono essere cattivi?

  %p Sì. Ecco alcuni modi in cui possono essere meno utili.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    Usare i commit log diffs al posto dei changelog è una brutta idea: contengono solo cose superflue.
    Cose come i merge commits, i commits con titoli oscuri,
    le modifiche della documentazione, etc.

  %p
    Lo scopo di un commit è quello di documentare un passo nell'evoluzione
    del codice sorgente. Alcuni progetti ripuliscono i commit, altri non lo fanno.

  %p
    Lo scopo di una voce changelog è quello di documentare le differenze rilevanti,
    spesso attraverso commit multipli,
    per comunicarli in modo chiaro agli utenti finali.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorare le deprecazioni

  %p
    Quando le persone aggiornano da una versione ad un'altra,
    dovrebbe essere dolorosamente chiaro che qualcosa si romperà.
    Dovrebbe essere possibile eseguire l'aggiornamento a una versione
    che elenca le deprecazioni, rimuovere ciò che è deprecato, quindi
    aggiornare alla versione in cui le deprecazioni diventano rimozioni.

  %p
    Se non fai nient'altro elenca le deprecazioni, le rimozioni e
    qualsiasi altro cambiamento nel tuo changelog.
  
  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Confusione delle date

  %p
    I formati di date variano da Paese a Paese e spesso
    trovare un formato human-friendly che sia intuitivo per tutti
    è cosa ardua. Il vantaggio delle date formattate come
    <code>2017-07-17</code> è che seguono l'ordine dal maggiore
    al minore: anno, mese e giorno. Inoltre questo formato non
    ha sovrapposizioni ambigue con altri formati di date, a differenza
    di alcuni formati regionali che scambiano la posizione del mese e del giorno.
    Queste motivazioni e il fatto che questo formato di data è uno
    #{link_to "standard ISO", data.links.isodate} spiegano perché questo è il formato di date
    raccomandato per i changelog.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Modifiche incoerenti

  %p
    Un changelog che menziona solo alcune delle modifiche può essere pericoloso
    tanto quanto non avere un changelog. Nonostante molte delle modifiche possano
    essere irrilevanti - per esempio, nella maggior parte dei casi rimuovere
    un singolo spazio non necessita di essere documentato - ogni modifica importante
    deve essere menzionata nel changelog. Applicando in maniera incoerente le modifiche,
    gli utenti possono erroneamente pensare che il changelog sia l'unica fonte
    di verità attendibile. E in realtà dovrebbe esserlo. Da un grande potere derivano
    grandi responsabilità - avere un buon changelog significa avere un changelog
    costantemente aggiornato.

  %aside
    C'è di più. Aiutatemi a collezionare altre situazioni cattive
    = link_to "aprendo un issue", data.links.issue
    o una pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Domande frequenti
  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Esiste un formato standard per i changelog?

  %p
    Non esattamente. Esistono le linee guida GNU sullo stile dei changelog, oppure
    i due lunghi paragrafi nel documento di GNU NEWS denominato "guideline". Entrambe
    sono inadeguate o insufficienti.

  %p
    Questo progetto vuole essere
    = link_to "una migliore convenzione per i file changelog.", data.links.changelog
    Per questo motivo osserviamo le migliori pratiche della comunità open source
    e le portiamo avanti.

  %p
    Critiche, discussioni e suggerimenti per migliorare
    = link_to "sono ben accetti.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Come si dovrebbe chiamare il file di changelog?

  %p
    Chiamalo <code>CHANGELOG.md</code>. Alcuni progetti usano anche
    <code>HISTORY</code>, <code>NEWS</code> o <code>RELEASES</code>.

  %p
    Risulta facile pensare che il nome del tuo file changelog
    non sia poi così importante, allora perché non rendere facile la
    vita ai tuoi utenti, usando sempre lo stesso nome?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Cosa dire delle GitHub Releases?

  %p
    È una bella iniziativa. #{link_to "Releases", data.links.github_releases} può essere usato
    per rendere semplice i git tags (per esempio il nome del tag <code>v1.0.0</code>)
    all'interno di note di rilascio ben dettagliate mediante l'aggiunta manuale delle note di rilascio
    oppure è possibile estrarre i messaggi annotati dei git tag e trasformarli in note.

  %p
    GitHub Releases crea un changelog non-portable che può essere solo
    visualizzato dagli utenti nel contesto di GitHub. È possibile farlo
    apparire molto simile al formato di Keep a Changelog, tuttavia tende
    ad essere un po' più complicato.

  %p
    La versione corrente di GitHub Releases risulta inoltre
    probabilmente poco rilevante per gli utenti finali, a differenza dei
    tipici file maiuscoli (<code>README</code>,
    <code>CONTRIBUTING</code>, etc.). Un altro problema minore è che
    l'interfaccia non offre attualmente link per la creazione di log tra
    ciascuna release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    I changelog possono essere analizzati automaticamente?

  %p
    È difficile, perché le persone usano formati e nomi di file
    profondamente diversi.

  %p
    #{link_to "Vandamme", data.links.vandamme} è una Ruby gem creata dal team
    Gemnasium ed è in grado di fare il parsing dei
    changelog di molti (ma non tutti) i progetti open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Cosa sono le release "yanked"?

  %p
    Le release "yanked" sono versioni che sono state rimosse a causa di
    bug seri o problemi di sicurezza. Spesso queste versioni non
    appaiono nei changelog. Invece dovrebbero esserci, nel seguente
    modo:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    L'etichetta <code>[YANKED]</code> è in maiuscolo per un motivo.
    È importante che le persone la notino.
    Poiché è racchiusa tra parentesi quadre è anche
    più semplice da riconoscere nel parsing automatico.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Si dovrebbe mai modificare un changelog?

  %p
    Certo. Ci sono sempre buoni motivi per migliorare un changelog. Io apro regolarmente
    delle pull request per aggiungere release mancanti ai progetti open source che non mantengono
    correttamente i changelog.

  %p
    È anche possibile che si scopra di aver dimenticato di annotare una modifica
    non retro-compatibile nelle note per una versione. Ovviamente è importante aggiornare
    il changelog in questo caso.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Come posso contribuire?

  %p
    Questo documento non è la <strong>verità assoluta</strong>; è solo la mia attenta
    opinione, arricchita dalle informazioni ed esempi che ho raccolto.

  %p
    Questo perché voglio che la nostra comunità raggiunga un consenso. Credo che
    la discussione sia importante almeno quanto il risultato finale.

  %p
    Quindi per favore <strong>#{link_to "partecipate", data.links.repo}</strong>.

.press
  %h3 Dialogo
  %p
    Sono andato a #{link_to "The Changelog podcast", data.links.thechangelog}
    per parlare del perché i gestori e i contributori dovrebbero preoccuparsi dei changelog
    e anche delle motivazioni dietro questo progetto.
```

## File: source/ja/1.0.0/index.html.haml
```haml
---
title: 変更履歴を記録する
description: 友達にGitログを変更履歴に移させないでください。
language: ja
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    変更履歴とは何ですか？

  %p
    変更履歴とは、プロジェクトの各バージョンに対する注目に値する変更点の時系列順に並べられたリストを含むファイルです。

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    なぜ変更履歴を記録するのですか？

  %p
    プロジェクトの各リリース（またはバージョン）の間で、どのような注目すべき変更が行われたのかをユーザーおよびコントリビューターが正確に把握しやすくするためです。

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    誰が変更履歴を必要としますか？

  %p
    人々です。消費者であろうと開発者であろうと、ソフトウェアのエンドユーザーはソフトウェアの内容を気にする人間です。ソフトウェアに変更があるとき、人々は変更の理由や方法を知りたいのです。

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    良い変更履歴を作るには？

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    基本理念

  %ul
    %li
      変更履歴は機械のためではなく<em>人間のため</em>にあります。
    %li
      バージョンごとに作成する必要があります。
    %li
      同じ種類の変更をグループ化する必要があります。
    %li
      バージョンとセクションはリンク可能である必要があります。
    %li
      最新バージョンは先頭にきます。
    %li
      各バージョンのリリース日を表示されます。
    %li
      #{link_to "Semantic Versioning", data.links.semver} に従っているかどうか言及してください。

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 変更の種類

  %ul
    %li
      %code Added
      新機能について。
    %li
      %code Changed
      既存機能の変更について。
    %li
      %code Deprecated
      間もなく削除される機能について。
    %li
      %code Removed
      今回で削除された機能について。
    %li
      %code Fixed
      バグ修正について。
    %li
      %code Security
      脆弱性に関する場合。

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    変更履歴のメンテナンスに必要な労力を減らすにはどうすればよいですか？

  %p
    今後の変更を追跡するには <code>Unreleased</code> セクションを上部に配置します。

  %p これには2つの目的があります。

  %ul
    %li
      人々は、今後のリリースでどのような変更が予想されるのかを確認することができます。
    %li
      リリース時には、 <code>Unreleased</code> セクションにある変更を
      新しいリリースバージョンのセクションに移動することができます。

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    変更履歴は悪くなる可能性がありますか？

  %p はい。いくつかの方法があります。

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    コミットログの差分

  %p
    変更履歴としてコミットログの差分を使用することはよくない考えです。それはノイズに満ちています。
    マージコミット、あいまいな表題のコミット、ドキュメントの変更などがあります。

  %p
    コミットの目的はソースコードの進化における一歩を文書化することです。
    コミットをクリーンアップするプロジェクトもあれば、そうでないプロジェクトもあります。

  %p
    変更履歴エントリの目的は、しばしば複数のコミットにまたがって注目すべき違いを文書化し、
    それらをエンドユーザーに明確に伝えることです。

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    非推奨の無視

  %p
    人々があるバージョンから別のバージョンにアップグレードするとき、いつ何かが壊れるのは痛いほど明らかです。
    廃止予定をリストアップしたバージョンにアップグレードし、廃止予定のものを削除してから、
    廃止予定が削除されるバージョンにアップグレードすることが可能です。

  %p
    あなたが他に何もしないのであれば、変更履歴に非推奨、削除、破壊的変更を列挙してください。


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    分かりにくい日付

  %p
    地域の日付形式は世界中で異なり、誰にとっても直感的でヒューマンフレンドリーな日付形式を見つけるのは困難です。
    <code>2017-07-17</code> のように形式化された日付の利点は、年、月、日のように最大から最小の単位の順序に従うということです。
    この形式は、月と日の位置を切り替える地域の形式とは異なり、他の日付形式とあいまいに重なり合うこともありません。
    これらの理由、およびこの日付形式が #{link_to "ISO standard", data.links.isodate} であるという事実が、それが変更履歴エントリに推奨される日付形式である理由です。

  %aside
    これだけではありません。
    = link_to "Issueを開く", data.links.issue
    か、プルリクエストにてこれらのアンチパターンの収集を手伝ってください。

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    よくある質問と回答

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    変更履歴の標準的な書式がありますか？

  %p
    実はありません。 #{link_to "GNU changelog style guide", data.links.gnustyle} 、
    もしくは #{link_to "two-paragraph-long GNU NEWS file", data.links.gnunews} "guideline" があります。
    どちらも不十分であり不適切です。

  %p
    このプロジェクトは
    = link_to "より良い変更履歴の規約", data.links.changelog
    になることを目指しています。
    それはオープンソースコミュニティの良い習慣を観察し、それらを集めることから来ます。

  %p
    健全な批判、議論、そして改善のための提案を
    = link_to "歓迎しています。", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    変更履歴ファイルにはどのような名前をつけるべきですか？

  %p
    <code>CHANGELOG.md</code> と呼びます。いくつかのプロジェクトでは
    <code>HISTORY</code> や <code>NEWS</code> 、 <code>RELEASES</code> が使われています。

  %p
    あなたの変更履歴ファイルの名前はそれほど重要でないと考えることは容易ですが、
    なぜエンドユーザーが一貫して注目の変更を見つけることを難しくするのですか？

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    GitHubリリースはどうですか？

  %p
    素晴らしい主導権です。 #{link_to "Releases", data.links.github_releases} は手動でリリースノートを追加することによって、
    単純なGitタグ(例えば <code>v1.0.0</code> という名前のタグ)をリッチなリリースノートに変換するために使用することができます。

  %p
    Githubリリースは、Githubのコンテキスト内でのみユーザーに表示できる移植性のない変更履歴を作成します。
    それらをKeep a Changelogの書式のように見せることは可能ですが、もう少し複雑になる傾向があります。

  %p
    Githubリリースの現在のバージョンも、典型的な大文字のファイル(<code>README</code> や <code>CONTRIBUTING</code>など)
    とは異なり、おそらくエンドユーザーにはあまり発見できません。
    もう一つの目立たない問題は、インターフェースが現在各リリース間のコミット履歴へのリンクを提供していないということです。

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    変更履歴を自動的に解析できますか？

  %p
    人々は大きく異なるフォーマットやファイル名に従うので、難しいです。

  %p
    #{link_to "Vandamme", data.links.vandamme} はGemnasiumチームによって作成されたRuby gemであり、
    （全てではないが）多くのオープンソースプロジェクトの変更履歴を解析します。


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    ヤンクリリースについてはどうですか？

  %p
    ヤンクリリースは、深刻なバグやセキュリティの問題のために
    引っ張られなければならなかったバージョンです。
    多くの場合、これらのバージョンは変更履歴に表示されません。表示しないべきである。
    次のように表示すべきである。

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> は大きな理由です。それに気付くことは人々にとって重要です。
    大括弧で囲まれているので、プログラムで解析するのも簡単です。


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    変更履歴を書き換える必要がありますか？

  %p
    もちろんです。変更履歴を改善するためには、常にもっともな理由があります。
    メンテナンスされていない変更履歴のあるオープンソースプロジェクトに
    不足しているリリースを追加するためのプルリクエストが常に開かれています。

  %p
    バージョンのノートにある破壊的変更への対応を忘れたことを発見するかもしれません。
    この場合、変更履歴を更新することは明らかに重要です。


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    どうやって貢献できますか？

  %p
    この文書は <strong>真実</strong> ではありません。
    私が集めた情報と例と共に、慎重に考えられた私の意見です。

  %p
    私たちのコミュニティが合意に達することを望んでいるからです。
    私は議論が最終結果と同じくらい重要であると思います。

  %p
    なので <strong>#{link_to "協力", data.links.repo}</strong> してください。

.press
  %h3 座談
  %p
    私は #{link_to "The Changelog podcast", data.links.thechangelog} で、メンテナーやコントリビューターがなぜ変更履歴を気にすべきなのか、
    そしてこのプロジェクトの背後にある動機について話しました。
```

## File: source/ja/1.1.0/index.html.haml
```haml
---
title: 変更履歴を記録する
description: 友達がGit記録を変更履歴に丸写しするのを止めさせよう。
language: ja
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    変更履歴とは何ですか？

  %p
    変更履歴とは、事業の各版に対する注目に値する変更点を
    時系列順に並べた一覧を含むファイルです。

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    なぜ変更履歴を記録するのですか？

  %p
    事業の各リリース（または各版）の間で、
    どのような注目すべき変更が行われたのかを利用者および貢献者が正確に把握しやすくするためです。

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    誰が変更履歴を必要としますか？

  %p
    全員が必要とします。
    消費者であろうと開発者であろうと、ソフトウェアの末端利用者はソフトウェアの内容を気にします。
    ソフトウェアに変更があるとき、人々は変更の理由や方法を知りたいのです。

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    良い変更履歴を作るには？

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    基本理念

  %ul
    %li
      変更履歴は機械のためではなく<em>人間のため</em>にあります。
    %li
      版ごとに作成する必要があります。
    %li
      同じ種類の変更をまとめる必要があります。
    %li
      版と節はリンク可能である必要があります。
    %li
      最新版は先頭にきます。
    %li
      各版のリリース日を表示されます。
    %li
      #{link_to "Semantic Versioning", data.links.semver} に従っているかどうか言及してください。

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 変更の種類

  %ul
    %li
      %code Added
      新機能について。
    %li
      %code Changed
      既存機能の変更について。
    %li
      %code Deprecated
      間もなく削除される機能について。
    %li
      %code Removed
      今回で削除された機能について。
    %li
      %code Fixed
      不具合修正について。
    %li
      %code Security
      脆弱性に関する場合。

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    変更履歴の維持管理に必要な労力を減らすにはどうすればよいですか？

  %p
    今後の変更を追跡するには <code>Unreleased</code> 節を上部に配置します。

  %p これには2つの目的があります。

  %ul
    %li
      人々は、今後のリリースでどのような変更が予想されるのかを確認することができます。
    %li
      リリース時には、 <code>Unreleased</code> 節にある変更を
      新しいリリース版の節に移動することができます。

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    変更履歴が害悪になる可能性はありますか？

  %p はい、ありえます。
  変更履歴を役立たずにしてしまう幾つかの行為をここに述べます。

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    コミット記録の差分

  %p
    変更履歴としてコミット記録の差分を使用することはよくない考えです。それは雑音に満ちています。
    コミット、あいまいな表題のコミット、文書の変更などがあります。

  %p
    コミットの目的はソースコードの進化における一歩を文書化することです。
    合併コミットを一掃する事業もあれば、そうでない事業もあります。

  %p
    変更履歴項目の目的は、しばしば複数のコミットにまたがって注目すべき違いを文書化し、
    それらを末端利用者に明確に伝えることです。

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    非推奨の無視

  %p
    人々がある版から別の版に増強するとき、いつ何かが壊れるのは痛いほど明らかです。
    廃止予定を洗い出した版に増強し、廃止予定のものを削除してから、
    廃止予定が削除される版に増強することが可能です。

  %p
    あなたが他に何もしないのであれば、変更履歴に非推奨、削除、破壊的変更を列挙してください。

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    分かりにくい日付

  %p
    地域の日付形式は世界中で異なり、
    誰にとっても直感的で親しみやすい日付形式を見つけるのは困難です。
    <code>2017-07-17</code> のように形式化された日付の利点は、
    年、月、日のように最大から最小の単位の順序に従うということです。
    この形式は、月と日の位置を切り替える地域の形式とは異なり、
    他の日付形式とあいまいに重なり合うこともありません。
    これらの理由、およびこの日付形式が #{link_to "国際標準", data.links.isodate} であるという事実が、
    それが変更履歴項目に推奨される日付形式である理由です。

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    一貫性のない変更

  %p
    変更履歴に全ての変更点を記載しないことは、変更履歴がないのと同じくらい危険です。
    たしかに記載しないでもよい変更は多くあります——
    例えば、空白を1つ削除したことはどんな場合でも記録する必要はないかもしれません——
    が、重要な変更は全実例において記載すべきです。
    変更を一貫して適用しないせいで、変更履歴こそ真実が記された唯一の情報源である
    という勘違いが生じる可能性があります。
    そして可能性は現実になりえます。
    大いなる力には大いなる責任が伴います——
    良い変更履歴を作るというのは更新が一貫した変更履歴を作るということを意味します。

  %aside
    これだけではありません。
    = link_to "Issueを開く", data.links.issue
    か、プルリクエストして、こういった反面教師の収集を手伝ってください。

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    よくある質問と回答

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    変更履歴の標準的な書式がありますか？

  %p
    実はありません。
    #{link_to "GNU changelog style guide", data.links.gnustyle} 、もしくは
    #{link_to "two-paragraph-long GNU NEWS file", data.links.gnunews} という
    「指針」があります。
    しかしどちらも不十分であり不適切です。

  %p
    この事業は
    = link_to "より良い変更履歴の規約", data.links.changelog
    になることを目指しています。
    それはオープンソース団体の良い習慣を観察し、それらを集めることから来ます。

  %p
    健全な批判、議論、そして改善のための提案を
    = link_to "歓迎しています。", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    変更履歴ファイルにはどのような名前をつけるべきですか？

  %p
    <code>CHANGELOG.md</code> という名前にしましょう。
    いくつかの事業では <code>HISTORY</code> や <code>NEWS</code> 、
    <code>RELEASES</code> という名前が使われています。

  %p
    あなたの変更履歴ファイルの名前はそれほど重要でないと考えることは容易ですが、
    しかし末端利用者が注目に値する変更を一貫して見つけることを難しくする理由はありません。

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    GitHubリリースはどうですか？

  %p
    極めて先駆的です。 #{link_to "Releases", data.links.github_releases} に手動でリリース告知を追加することによって、
    単純なGitタグ（例えば <code>v1.0.0</code> という名前の標識）を素敵なリリース告知に変換するために使用できます。

  %p
    GitHub Releasesは、GitHubの文脈内でのみ利用者に表示できる移植性のない変更履歴を作成します。
    それらをKeep a Changelogの書式のように見せることは可能ですが、もう少し複雑になる傾向があります。

  %p
    GitHub Releasesの現在の版も、典型的な大文字のファイル
    （<code>README</code> や <code>CONTRIBUTING</code>など）
    とは異なり、おそらく末端利用者が見付けるのは簡単ではないでしょう。
    もう一つ目立たない問題として、現在、各リリース間のコミット履歴への
    リンクが提供されていないということがあります。

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    変更履歴を自動的に解析できますか？

  %p
    人々によって形式やファイル名は大きく異なるため、難しいです。

  %p
    #{link_to "Vandamme", data.links.vandamme} はGemnasiumチームが作成したRuby gemであり、
    （全てではありませんが）多くのオープンソース事業の変更履歴を解析できます。


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    リリース引き戻しについてはどうですか？

  %p
    リリース引き戻し (yanked releases) とは、深刻な不具合や安全上の問題のために
    リリースを引き戻さ (yank) なれなければならなかった版です。
    これらの変更はしばしば変更履歴に記載さえされませんが，必ず記載すべきです。
    次のように表示すればよいでしょう。

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> が仰々しいのには訳があります。
    利用者が引き戻しに気付くことが重要だからです。
    角括弧で囲まれているので、プログラムで解析するのも簡単です。


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    変更履歴を書き換える必要がありますか？

  %p
    もちろんです。変更履歴を改善するためには、常にもっともな理由があります。
    維持管理されていない変更履歴のあるオープンソース事業に、
    不足している資源を追加するためのプルリクエストが常に開かれています。

  %p
    版の告知にある破壊的変更への対応を忘れたことを発見するかもしれません。
    この場合、変更履歴を更新することは明らかに重要です。

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    どうやって貢献できますか？

  %p
    この文書は <strong>真実</strong> ではありません。
    私が集めた情報と例と共に、慎重に考えられた私の意見です。

  %p
    私たち一同が合意に達することを望んでいるからです。
    私は議論が最終結果と同じくらい重要であると思います。

  %p
    なので <strong>#{link_to "協力", data.links.repo}</strong> してください。

.press
  %h3 座談
  %p
    私は #{link_to "The Changelog podcast", data.links.thechangelog} で、
    管理者や貢献者がなぜ変更履歴を気にすべきなのか、
    そしてこの事業の背後にある動機について話しました。
```

## File: source/ka/1.0.0/index.html.haml
```haml
---
title: შეინარჩუნე ცვლილებების ჟურნალი
description: არ მისცე უფლება მეგობრებს git ჩანაწერები ჩაწერონ ცვლილებების ჟურნალში
language: en
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    რა არის ცვლილებების ჟურნალი (changelog)?

  %p
    ცვლილებების ჟურნალი არის ფაილი რომელიც შეიცავს ქრონოლოგიურად დალაგებულ და კურირებულ სიას
    იმ აღწერებისა რაც პროექტის თითოეულ ვერსიას აქვს

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    რატომ უნდა მოვუფრთხილდეთ ცვილებების ჟურნალს?

  %p
    რათა მომხმარებლებმა და კონტრიბუტორებმა ნათლად დაინახონ რა შეიცვალა პროექტის გამოშვებებს (ან ვერსიებს) შორის.
    

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    ვის სჭირდება ცვლილებების ჟურნალი?

  %p
    ხალხს სჭირდება. მნიშვნელობა არ აქვს მომხმარებელი იქნება თუ დეველოპერი, საბოლოო ჯამში ადამიანები არიან ისინი ვინც
    ზრუნავენ რა არის ამ პროგრამაში. როდესაც პროგრამა იცვლება, ხალხს უნდა იცოდეს რატომ და როგორ.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    როგორ გავაკეთო კარგი ცვლილებების ჟურნალი?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    მიყევი პრინციპებს

  %ul
    %li
      ცვლილებების ჟურნალი <em>განკუთვნილია ადამიანებისთვის</em>, არა მანქანებისთვის.
    %li
      თითოეული ვერსიისთვის უნდა არსებობდეს ჩანაწერი.
    %li
      ერთნაირი ტიპის ჩანაწერები უნდა დაჯგფუდეს.
    %li
      ვერსიები და სექციების გადაბმა შესაძლებელი უნდა იყოს.
    %li
      უკანასკნელი ვერსია უნდა ეწეროს პირველი.
    %li
      გამოშვების თარიღი უნდა ეწეროს თითოეულ ვერსიას.
    %li
      ახსენე თუ იყენებ #{link_to "სემანტიკურ ვერსიონირებას", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types ცვლილებების ტიპი

  %ul
    %li
      %code Added
      ახალი ფუნქციონალისთვის.
    %li
      %code Changed
      არსებულ ფუნქციონალში შეტანილი ცვლილებებისთვის.
    %li
      %code Deprecated
      მალე წასაშლელი ფუნქციონალისთვის.
    %li
      %code Removed
      უკვე წაშლილი ფუნქციონალისთვის.
    %li
      %code Fixed
      შეცდომების გასწორებისთვის.
    %li
      %code Security
      მოწყვლადობის აღმოჩენის შემთხვევაში.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    როგორ შევამცირო ძალისხმევა ცვლილებების ჟურნალის შესანარჩუნებლად?

  %p
    <code>Unreleased</code> სექცია დატოვე ყოველთვის მაღლა რათა მარტივად შეიძლებოდეს დანახვა მომავალი ცვლილებების

  %p ამას ორი მიზეზი აქვს:

  %ul
    %li
      ხალხს შეეძლება დაინახოს რა მოსალოდნელი ცვლილებებია მომავალ გამოშვებებში
    %li
      გაშვების დროს შეგიძლია უბრალოდ <code>Unreleased</code> სექცია შეცვალო ახალი ვერსიის სექციად.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    შესაძლებელია ცვლილებების ჟურნალი იყოს ცუდი?

  %p დიახ. აქ არის რამოდენიმე მაგალითი რომელიც ნათლად გვაჩვენებს ამას.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    commit-ების ჩანაწერების სხვაობა.


  %p
    commit-ების ჩანაწერების სხვაობის გამოყენება ცვლილებების ჟურნალის სახით ცუდი იდეაა: 
    სავსეა უსარგებლო ინფორმაციით, მათ შორის: არასწორი სახელებით და დოკუმენტაციის ცვლილებებით.

  %p
    commit-ის დანიშნულება არის კოდის ევოლუციის აღწერა. ზოგიერთი პროექტი შლის მათ, ზოგიერთი არა.
    
  %p
    ცვლილებების ჟურნალში ჩანაწერის მიზანი არის მნიშვნელოვანი ცვლილებები აღწეროს, ხშირად რამოდენიმე კომიტის
    ერთად, რათა მარტივად იყოს საბოლოო მომხმარებლებისთვის გასაგები.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    მოძველებული ფუნქციონალის იგნორირება

  %p
    როდესაც ხალხი ერთი ვერსიიდან განახლებას აკეთებს შემდგომზე, აშკარა უნდა იყოს რა შეიძლება გაფუჭდეს.
    ჯერ განაახლე ისეთ ვერსიაზე რომელსაც აქვს მოძველებული ფუნქციონალის სია, წაშალე რაც მოძველდა და შემდგომ
    ისეთ ვერსიაზე განაახლე სადაც მოძველებული ფუნქციონალი უკვე წაშლილია. 

  %p
    თუ სხვა არაფერს აკეთებ, ჩამოწერე რაც მოძველდა, რაც წაიშალა და ნებისმიერი სხვა ცვლილება ამ ჟურნალში.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    დამაბნეველი თარიღები

  %p
    რეგიონალური თარიღის ფორმატები განსხვავდება მთელი მსოფლიოს მასშტაბით 
    და ხშირად რთულია იპოვო ადამიანისთვის მარტივად გასაგები თარირის ფორმატი, 
    რომელიც ინტუიციურია ყველასთვის. <code>2017-07-17</code> ასეთი ფორმატების უპირატესობა
    არის ის, რომ მიმდევრობას ემორჩილება უდიდესიდან უმცირეს საზომ ერთეულამდე:
    წელი, თვე და დღე. ეს ფორმატი ასევე არ ემთხვევა სხვა თარიღის ფორმატებს უცნაური ვარიანტებით,
    განსხვავებით სხვა ფორმატებისგან რომლებიც ზოგჯერ ცვლიან დღის და თვის რიცხვების პოზიციას.
    ამ მიზეზთან გამო და იმის გამო, რომ ეს ფორმატი 
    #{link_to "ISO სტანდარტია", data.links.isodate}, იგი რეკომენდირებული თარიღის ფორმატია 
    ცვლილებების ჟურნალის ჩანაწერებისთვის.

  %aside
    დამეხმარე სხვა არაზუსტი შაბლონების შეგროვებაში
    = link_to "ახალი საკითხის გახსნით", data.links.issue
    ან pull მოთხოვნით.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    ხშირად დასმული კითხვები

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    არის თუ არა რაიმე სტანდარტული ფორმატი ცვლილებების ჟურნალისთვის?

  %p
    არა. არსებობს #{link_to "GNU ცვლილებების ჟურნალის სტილი", data.links.gnustyle},
    ან #{link_to "2 პარაგრაფის სიგრძის GNU NEWS ფაილი", data.links.gnunews}
    "დირექტივა". ორივე შეუფერებელი და არასაკმარისია.

  %p
    პროექტის მიზანია
    = link_to "ჩამოაყალიბოს უკეთესი კონვენცია ცვლილებების ჟურნალისთვის.", data.links.changelog
    რომელიც შედეგია წარმატებულ გამოცდილებებზე დაკვირვების ჩვენს კომუნაში და შემდგომ ერთად თავმოყრის.

  %p
    ჯანსაღი კრიტიკა, დისკუსია და გაუმჯობესების რჩევები 
    = link_to "მისასალმებელია.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    რა უნდა დაერქვას ცვლილებების ჟურნალის ფაილს?

  %p
    დაარქვი <code>CHANGELOG.md</code>. ზოგი პროექტი იყენებს
    <code>HISTORY</code>, <code>NEWS</code> ან <code>RELEASES</code> სახელს.

  %p
    ადვილია იფიქრო რომ ჟურნალის ფაილის სახელს დიდი მნიშვნელობა არ აქვს,
    მაგრამ რატომ უნდა გაურთულო შენი პროგრამის მომხმარებლებს ცვლილებების ძიება?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    რა ხდება GitHub Releases დროს?

  %p
    შესანიშნავი ინიციატივაა. #{link_to "Releases", data.links.github_releases} შესაძლებელია გამოყენებულ იქნას როგორც,
    უბრალო git იარლიყების(tag) (მაგალითად იარლიყი სახელად <code>v1.0.0</code>)
    გადასაქცევად სრულყოფილ გამოშვების ჩანაწერებად ჩვენს მიერ
    ან შესაძლებელია git იარლიყების შეტყობინების ავტომატურად წამოღება და ჩანაწერად ქცევა.
    
  %p
    GitHub Releases ქმნის არაპორტაბელურ ცვლილებების ჟურნალს რომელიც
    მხოლოდ GitHub-ის შიგნით შეიძლება იქნას გამოყენებული. შესაძლებელია მათი
    წარმოჩენა სხვა ფორმატით თუმცა უფრო მეტი შრომა სჭირდება.

  %p
    GitHub Releases მიმდინარე ვერსია არ არის მარტივად საპოვნელი მომხმარებლებისთვის, განსხავებით
    სხვა ტიპიური ფაილებისგან (<code>README</code>, <code>CONTRIBUTING</code> და სხვა.). ასევე ინტერფეისი
    ამ ეტაპზე არ იძლევა საშუალებას გვქონდეს ბმული commit ჩანაწერებზე თითოეულ გამოშვებას შორის.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    შესაძლებელია ცვლილებების ჟურნალის ავტომატური სინტაქსური ანალიზი?

  %p
    რთულია, რადგან ხალხი იყენებს რადიკალურად განსხვავებულ ფორმატებს და ფაილის სახელებს.
    
  %p
    #{link_to "Vandamme", data.links.vandamme} არის Ruby gem შექმნილი
    Gemnasium გუნდის მიერ რომელიც აანალიზებს უმრავლეს (თუმცა 
    ყველას არა) ,ღია პროექტის, ცვლილებების ჟურნალს.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    რა ხდება yanked გამოშვებებისას?

  %p
    Yanked გამოშვებები არის ისეთი ვერსიები რომლებიც უკან უნდა დაბრუნდეს სერიოზული
    შეცდომის ან უსაფრთხოების პრობლემის გამო. ხშირად ასეთი ვერსიები არც ხვდება ცვლილებების ჟურნალში.
    არადა უნდა გამოჩნდეს და ასე მაგალითად:
    
  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> იარლიყი სპეციალურად არის. მნიშვნელოვანია რომ შეამჩნიოს ხალხმა.
    ასევე რადგანაც ის კვადრატულ ფრჩხილებშია მოქცეული, პროგრამულადაც ადვილია სინტაქსური დამუშავება.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    უნდა გადაწერო თუ არა როდისმე ცვლილებების ჟურნალი?

  %p
    რათქმაუნდა. ყოველთვის არის მიზეზი რათა გააუმჯობესო ჟურნალი.
    რეგულარულად გახსენი pull მოთხოვნები რათა დაამატო გამოტოვებული გამოშვებები
    სხვადასხვა ღია პროექტების უპატრონოდ მიტოვებულ ცვლილებების ჟურნალს.
    

  %p
    ასევე შესაძლებელია აღმოაჩინო რომ დაგავიწყდა რომელიმე ისეთი ცვლილების ჩაწერა კონკრეტული ვერსიისთვის,
    რაც აფუჭებს პროგრამას. ამ შემთხვევაში აშკარაზე აშკარაა რომ განაახლო შენი ცვლილებების ჟურნალი.
    

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    როგორ შემიძლია კონტრიბუცია?

  %p
    ეს დოკუმენტი არ არის აბსოლუტური <strong>ჭეშმარიტება</strong>; ეს უფრო არის ჩემს მიერ გათვალისწინებული მოსაზრებები
    სხვადასხვა მაგალითებთან და მოპოვებულ ინფორმაციასთან ერთად.

  %p
    მინდა რომ ჩვენმა კომუნამ მიაღწიოს კონსესუსს და მჯერა რომ ეს დისკუსია ისეთივე მნიშვნელოვანია როგორც საბოლოო შედეგი.
    

  %p
    ასე რომ გთხოვ <strong>#{link_to "ჩაერთე", data.links.repo}</strong>.

.press
  %h3 საუბრები
  %p
    ვიყავი #{link_to "The Changelog პოდკასტის", data.links.thechangelog} სტუმარი და ვისაუბრე
    რატომ უნდა იზრუნონ ცვლილებების ჟურნალზე კონტრიბუტორებმა და ისინი ვინც კოდს ინარჩუნებენ ყველაზე მეტად.
    ამავდროულად მოტივაციაზე ამ პროექტის უკან.
```

## File: source/ko/1.0.0/index.html.haml
```haml
---
title: Keep a Changelog
description: 동료가 git 로그를 changelogs에 덤프하게 내버려 두지 마세요.
language: ko
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Changelog는 무엇인가요?

  %p
    Changelog는 프로젝트의 각 버전에 대해 선별된 눈에 띄는 변경사항을 시간 순서대로 정리해둔 파일입니다.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    왜 changelog를 유지해야 하나요?

  %p
    사용자와 기여자가 프로젝트의 각 릴리즈(또는 버전)간에 정확히 어떤 주목할만한 변경사항이 있는지 보기 쉽도록 합니다.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    누가 changelog를 필요로 하나요?

  %p
    사람들이 필요로 합니다. 개발자이든 사용자이든, 소프트웨어의 최종 사용자는 소프트웨어에 무엇이 있는지 관심이 있는 사람입니다.
    소프트웨어가 변할 때, 사람들은 왜 그리고 어떻게 바뀌었는지 알고 싶어합니다.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    어떻게 좋은 changelog를 만들수 있나요?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    가이드 원칙

  %ul
    %li
      Changelogs는 <em>사람을 위한 것입니다.</em> 기계를 위한 것이 아닙니다.
    %li
      모든 단일 버전에 대한 항목이 있어야 합니다.
    %li
      같은 유형의 변경사항은 모아야 합니다.
    %li
      버전과 섹션은 연결할 수 있어야 합니다.
    %li
      최신 버전이 처음에 나옵니다.
    %li
      각 버전의 릴리즈 날짜를 표시해야 합니다.
    %li
      #{link_to "시멘틱 버저닝", data.links.semver}를 따르는지 언급해 주세요.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 변경 유형

  %ul
    %li
      %code Added
      새로운 기능
    %li
      %code Changed
      기존 기능의 변경사항
    %li
      %code Deprecated
      곧 지워질 기능
    %li
      %code Removed
      지금 지워진 기능
    %li
      %code Fixed
      버그 픽스
    %li
      %code Security
      취약점이 있는 경우

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    changelog를 관리하는 노력을 어떻게 줄일 수 있나요?

  %p
    <code>Unreleased</code> 섹션을 가장 위에 두어 다가올 변경사항을 추적할 수 있도록 하세요.

  %p 이것은 두 가지 용도로 사용됩니다:

  %ul
    %li
      사람들이 다음 릴리즈에서 기대하는 변경사항을 확인할 수 있습니다.
    %li
      릴리즈 시점에 <code>Unreleased</code> 섹션의 변경사항을 새 릴리즈 섹션으로 이동할 수 있습니다.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    changelogs가 안좋게 될 수 있습니까?

  %p 네. 여기에 changelog가 쓸모없게 되는 몇가지 경우들이 있습니다.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    커밋 로그 차이(Commit log diffs)

  %p
    커밋 로그 차이를 changelog로 사용하는 것은 안좋은 생각입니다:
    머지 커밋, 모호한 타이틀을 가진 커밋, 문서 변경 등 노이즈로 가득차 있습니다.

  %p
    커밋의 목적은 소스 코드 진화의 단계를 기록하기 위함입니다.
    어떤 프로젝트는 커밋을 정리하지만, 어떤 프로젝트는 하지 않습니다.

  %p
    changelog 기입의 목적은 종종 다수의 커밋 중에서 주목할만한 차이를
    최종 사용자에게 명확하게 전달하기 위해 문서화하는 것입니다.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    없어질 기능들(Deprecations) 무시하기

  %p
    사람들이 다른 버전으로 업그레이드 할 때, 언제 어떤 것이 손상될수있는지(breakable) 고통스럽게 분명해야 합니다.
    앞으로 사라질 기능들(deprecations)이 나열된 버전으로 업그레이드하고,
    더 이상 사용하지 않는 것(deprecated)을 제거한 뒤, 그 사라질 기능들이
    정말 없어진 버전으로 업데이트 하는 것이 가능해야 합니다.

  %p
    아무 작업도 수행하지 않는다면, 없어질 기능들, 제거된 것, 모든 급격한 변화를 changelog에 남기십시오.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    날짜를 혼동하는 것

  %p
    지역 날짜 포맷은 전세계에 걸쳐 다르고 종종 모두에게 직관적인 인간 친화적 날짜 포맷을 찾기 힘듭니다.
    <code>2017-07-17</code> 같은 날짜 포맷(연, 월, 일)의 장점은
    큰 단위부터 작은 단위의 순서를 따른다는 것입니다.
    월과 일의 위치가 뒤바뀐 어떤 포맷과 다르게, 이 포맷은 다른 날짜 포맷과 모호하게 겹치는 부분이 없습니다.
    이런 이유와 이 포맷이 #{link_to "ISO standard", data.links.isodate}라는 사실이
    changelog 기입을 위해 이 날짜 포맷을 추천하는 이유입니다.

  %aside
    안티패턴은 더 있습니다.
    = link_to "이슈 오픈하기", data.links.issue
    나 pull 요청을 통해
    안티패턴들을 모으는 것을 도와주세요.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    자주 하는 질문

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    changelog의 표준 포맷이 있나요?

  %p
    없습니다. GNU changelog 스타일 가이드나 두 문단정도의 GNU NEWS '가이드라인'이 있습니다.
    하지만 둘다 부적절하거나 충분하지 않습니다.

  %p
    이 프로젝트의 목표는
    = link_to "더 나은 changelog 규칙", data.links.changelog
    입니다.
    이것은 오픈소스 커뮤니티에서 좋은 용례를 관찰하고 모으는데서 나옵니다.

  %p
    건강한 비판, 토론 및 개선 제안은
    = link_to "환영합니다.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    changelog 파일의 이름을 무엇으로 지어야 하나요?

  %p
    <code>CHANGELOG.md</code>라고 만드세요. 어떤 프로젝트는
    <code>HISTORY</code>, <code>NEWS</code> 또는 <code>RELEASES</code>를 사용합니다.

  %p
    changelog 파일의 이름이 무슨 상관이냐고 생각하기 쉽겠지만,
    왜 굳이 여러분의 사용자가 변경사항을 일관적으로 찾기 힘들도록 만드나요?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    깃허브 릴리즈는 어떻게 하나요?

  %p
    이것은 훌륭한 이니셔티브입니다. #{link_to "릴리즈", data.links.github_releases}는
    직접 릴리즈 노트를 추가하거나 어노테이션된 깃 태그 메시지를 가져와서 노트로 바꿔
    간단한 깃 태그(예를 들어 <code>v1.0.0</code> 태그
    )를 풍부한 릴리즈 노트로 전환시키는데 사용될 수 있습니다.

  %p
    깃허브 릴리즈는 이동 불가능한 깃허브 컨텍스트 내에서만 표시되는 changelog를 생성합니다.
    Keep a Changelog 포맷처럼 보이게 만드는 게 가능하지만, 좀 더 복잡해지는 경향이 있습니다.

  %p
    전형적인 대문자 파일들과 달리(<code>README</code>, <code>CONTRIBUTING</code>, 등),
    깃허브 릴리즈의 현재 버전은 최종 사용자가 거의 찾아볼 수 없습니다.
    다른 사소한 이슈는 인터페이스가 현재 각 릴리즈 사이에 로그를 커밋할 수 있는 링크를 제공하지 않는 것입니다.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Changelog를 자동으로 파싱할 수 있나요?

  %p
    사람들이 대단히 다양한 포맷과 파일 이름을 따르기 때문에 어렵습니다.

  %p
    #{link_to "Vandamme", data.links.vandamme}은 Gemnasium 팀에 의해
    생성된 루비잼이고 많은(전부는 아니고) 오픈소스 프로젝트의 changelog를 파싱합니다.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    삭제된 릴리즈(Yanked release)는 어떻게 하나요?

  %p
    삭제된(Yanked) 릴리즈는 심각한 버그나 보안 이슈 때문에 소스에서 들어내버린 버전을 말합니다.
    대게 이런 버전은 changelog에 아예 나오지도 않지만, 나와야 합니다. 이것이 삭제된 릴리즈를
    표시하는 방법입니다:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> 태그가 요란한 이유가 있습니다.
    사람들이 알아차리는 것이 중요합니다. 대괄호 안에 있기 때문에 프로그래밍적으로 파싱하기에도 용이합니다.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    changelog를 다시 써야하나요?

  %p
    물론입니다. changelog를 개선할 좋은 이유는 항상 있습니다. 저는 정기적으로
    changelog가 관리되지 않는 오픈소스에 빠진 릴리즈를 추가하기 위해 pull request를 오픈합니다.

  %p
    어떤 버전의 급격한 변화에 대해 언급하는 것을 잊은 것을 발견할 수도 있습니다.
    이 경우엔 changelog를 업데이트하는 것이 당연히 중요합니다.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    어떻게 기여할 수 있나요?

  %p
    이 문서가 <strong>진리</strong>는 아닙니다. 이것은 제가 모은 정보와 예제들과 함께
    신중하게 고려한 의견입니다.

  %p
    왜냐하면 우리 커뮤니티가 공감대를 형성하기를 원하기 때문입니다. 저는 최종결과 못지않게 토론도 중요하다고 생각합니다.

  %p
    그러니 <strong>#{link_to "참여", data.links.repo}</strong>를 부탁합니다.

.press
  %h3 대화
  %p
    왜 관리자와 기여자가 changelog를 신경써야하는지, 또한 이 프로젝트를 하게된 동기에 대해 이야기하기 위해
    #{link_to "The Changelog 팟캐스트", data.links.thechangelog}에 다녀왔습니다.
```

## File: source/ko/1.1.0/index.html.haml
```haml
---
title: Keep a Changelog
description: 동료들이 git 로그를 그대로 changelogs에 올리게 내버려 두지 마세요.
language: ko
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Changelog는 무엇인가요?

  %p
    Changelog는 프로젝트의 각 버전에 대해 주목할 만한 변경 사항들을 시간 순서대로 정리한 파일입니다.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    왜 changelog를 유지해야 하나요?

  %p
    사용자와 기여자가 프로젝트의 각 릴리즈(또는 버전)간에 정확히 어떤 주목할만한 변경 사항이 있는지 알기 쉬워집니다.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    누가 changelog를 필요로 하나요?

  %p
    사람들이 필요로 합니다. 개발자이든 사용자이든, 소프트웨어의 최종 사용자는 소프트웨어에 무엇이 있는지 관심이 있는 사람입니다.
    소프트웨어에 변경 사항이 있을 때, 사람들은 왜 그리고 어떻게 바뀌었는지 알고 싶어 합니다.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    어떻게 좋은 changelog를 만들수 있나요?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    가이드 원칙

  %ul
    %li
      Changelogs는 <em>사람을 위한 것입니다.</em> 기계를 위한 것이 아닙니다.
    %li
      모든 단일 버전에 대한 항목이 있어야 합니다.
    %li
      같은 유형의 변경 사항은 그룹으로 묶어야 합니다.
    %li
      버전 및 섹션들은 링크를 통해 연결될 수 있어야 합니다.
    %li
      최신 버전이 처음에 와야 합니다.
    %li
      각 버전의 릴리즈 날짜를 표시해야 합니다.
    %li
      프로젝트가 #{link_to "시맨틱 버저닝", data.links.semver}을 따르는지를 언급해야 합니다.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 변경 유형

  %ul
    %li
      %code Added
      새로운 기능
    %li
      %code Changed
      기존 기능의 변경
    %li
      %code Deprecated
      곧 없어질 기능
    %li
      %code Removed
      없어진 기능
    %li
      %code Fixed
      버그 수정
    %li
      %code Security
      보안 취약점에 관하여

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Changelog를 관리하기 위해 필요한 노력을 어떻게 줄일 수 있나요?

  %p
    <code>Unreleased</code> 섹션을 가장 위에 두어 다가올 변경 사항들을 추적할 수 있도록 하세요.

  %p 이것은 두 가지 목적이 있습니다.

  %ul
    %li
      사람들이 향후 릴리즈에서 예상되는 변경 사항들을 확인할 수 있습니다.
    %li
      릴리즈 시점에 <code>Unreleased</code> 섹션의 변경 사항들을 새 릴리즈 버전의 섹션으로 이동할 수 있습니다.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Changelogs가 나빠질 수 있나요?

  %p 네. 여기에 changelog가 쓸모없게 되는 몇 가지 경우들이 있습니다.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    커밋 로그 차이(Commit log diffs)

  %p
    커밋 로그의 차이를 changelog로 사용하는 것은 안 좋은 생각입니다.
    커밋 로그는 머지 커밋, 모호한 타이틀을 가진 커밋, 문서 변경 등 노이즈로 가득 차 있습니다.

  %p
    커밋의 목적은 소스 코드 진화의 단계를 기록하기 위함입니다.
    어떤 프로젝트는 커밋을 정리하지만, 어떤 프로젝트는 하지 않습니다.

  %p
    changelog 기재의 목적은 흔히 다수의 커밋에 걸친 주목할만한 차이를
    최종 사용자에게 명확하게 전달하기 위해 문서화하는 것입니다.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    없어질 기능들(Deprecations) 무시하기

  %p
    사람들이 다른 버전으로 업데이트 할 때, 언제 어떤 것이 중단될지(break) 극도로 분명해야 합니다.
    앞으로 없어질 기능들(deprecations)이 나열된 버전으로 업데이트하고,
    더 이상 사용하지 않는 기능(deprecated)을 제거한 뒤, 그 기능들이
    정말 삭제된 버전으로 업데이트 하는 것이 가능해야 합니다.

  %p
    최소한, 앞으로 없어질 기능들, 제거된 것 그리고 모든 급격한 변화(breaking changes)는 changelog에 남기십시오.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    혼란스러운 날짜

  %p
    지역별 날짜 포맷은 전세계에 걸쳐 다양하며 모든 사람이 직관적으로 느낄 수 있는 인간 친화적 날짜 포맷을 찾기 힘든 경우가 많습니다.
    <code>2017-07-17</code> 같은 날짜 포맷(연, 월, 일)의 장점은
    큰 단위부터 작은 단위의 순서를 따른다는 것입니다.
    월과 일의 위치가 뒤바뀐 어떤 포맷과 다르게, 이 포맷은 다른 날짜 포맷과 모호하게 겹치는 부분이 없습니다.
    이런 이유와 이 포맷이 #{link_to "ISO standard", data.links.isodate}라는 사실이
    changelog 기재에 이 날짜 포맷을 추천하는 이유입니다.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    일관성 없는 변경 사항

  %p
    일부 변경 사항만 언급하는 changelog는 changelog가 없는 것만큼 위험할 수 있습니다.
    물론 기록하지 않아도 되는 변경 사항도 많이 있습니다.
    예를 들어, 단순히 공백 하나를 삭제한 것은 어떤 경우에도 기록할 필요가 없을 수도 있습니다.
    하지만 중요한 변경 사항은 changelog에 명시되어야 합니다.
    일관성 없이 변경 사항을 적용하면 사용자들이 changelog가 단일 진실 공급원(Single source of truth)이라고
    착각할 수 있습니다.
    큰 권한에는 큰 책임이 따릅니다. 좋은 변경 로그는 일관성 있게 업데이트되는 변경 로그를 의미합니다.

  %aside
    안티패턴은 더 있습니다.
    = link_to "이슈 오픈하기", data.links.issue
    나 pull request를 통해
    안티패턴들을 모으는 것을 도와주세요.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    자주 하는 질문

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Changelog의 표준 포맷이 있나요?

  %p
    없습니다. #{link_to "GNU changelog 스타일 가이드", data.links.gnustyle} 또는
    두 문단 정도의 #{link_to "GNU NEWS file", data.links.gnunews} "가이드라인"이 있습니다.
    하지만 둘 다 부적절하거나 불충분합니다.

  %p
    이 프로젝트의 목표는
    = link_to "더 나은 changelog 규칙", data.links.changelog
    입니다.
    이것은 오픈소스 커뮤니티에서 좋은 사례를 관찰하고 모으는 데서 비롯됩니다.

  %p
    건강한 비판, 토론 및 개선 제안은
    = link_to "환영합니다.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Changelog 파일의 이름을 무엇으로 지어야 하나요?

  %p
    <code>CHANGELOG.md</code>라고 만드세요. 어떤 프로젝트는
    <code>HISTORY</code>, <code>NEWS</code> 또는 <code>RELEASES</code>를 사용합니다.

  %p
    changelog 파일의 이름이 무슨 상관이냐고 생각하기 쉽겠지만,
    최종 사용자가 주목할 만한 변경 사항을 일관적으로 찾기 쉬워집니다.

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    깃허브 릴리즈는 어떻게 하나요?

  %p
    이것은 훌륭한 계획입니다. #{link_to "릴리즈", data.links.github_releases}는
    직접 릴리즈 노트를 추가하여 간단한 깃 태그(예를 들어 <code>v1.0.0</code> 태그)를
    풍부한 릴리즈 노트로 전환하거나, 주석이 달린 깃 태그 메시지를 가져와서
    릴리즈 노트로 전환할 수 있습니다.

  %p
    깃허브 릴리즈는 이식 불가능한 changelog를 생성하며, 이는 깃허브 내에서만 사용자들에게 보여집니다.
    Keep a Changelog 포맷처럼 보이게 만드는 게 가능하지만, 좀 더 복잡해지는 경향이 있습니다.

  %p
    일반적인 대문자 파일들(<code>README</code>, <code>CONTRIBUTING</code> 등)과 달리,
    깃허브 릴리즈의 현재 버전은 최종 사용자가 쉽게 찾아볼 수 없습니다.
    또 다른 사소한 이슈는 깃허브 릴리즈의 인터페이스가 현재 각 릴리스 간의 커밋 로그 링크를 제공하지 않는다는 것입니다.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Changelog를 자동으로 파싱할 수 있나요?

  %p
    사람들이 매우 다양한 포맷과 파일 이름을 따르기 때문에 어렵습니다.

  %p
    #{link_to "Vandamme", data.links.vandamme}은 Gemnasium 팀에 의해
    생성된 Ruby gem이고 전부는 아니지만 많은 오픈소스 프로젝트의 changelog를 파싱합니다.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    삭제된 릴리즈(Yanked release)는 어떻게 하나요?

  %p
    삭제된(Yanked) 릴리즈는 심각한 버그나 보안 이슈 때문에 소스에서 들어내 버린 버전을 말합니다.
    이런 버전은 changelog에 나타내지 않는 경우가 않지만, 나타내야 합니다. 이것이 삭제된 릴리즈를
    표시하는 방법입니다.

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    사람들이 알아차리는 것이 중요하기 때문에 눈에 띄는 <code>[YANKED]</code> 태그가 있습니다.
    대괄호 안에 있기 때문에 프로그래밍적으로 파싱하기에도 용이합니다.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Changelog를 다시 작성한 적이 있나요?

  %p
    물론입니다. changelog를 개선할 좋은 이유는 항상 있습니다. 
    저는 정기적으로 changelog가 관리되지 않은 오픈소스에 pull request를 열어서
    누락된 릴리즈를 추가합니다.

  %p
    또한, 어떤 버전의 급격한 변화에 대한 언급을 잊은 걸 발견할 수도 있습니다.
    이 경우엔 changelog를 업데이트하는 것이 당연히 중요합니다.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    어떻게 기여할 수 있나요?

  %p
    이 문서가 <strong>진리</strong>는 아닙니다. 이것은 제가 모은 정보와 예제들과 함께
    신중하게 고려한 의견입니다.

  %p
    이는 우리 커뮤니티가 합의점에 도달하기를 원하기 때문입니다. 저는 최종결과 못지않게 토론도 중요하다고 생각합니다.

  %p
    그러니 많은 <strong>#{link_to "참여", data.links.repo}</strong> 부탁드립니다.

.press
  %h3 담화
  %p
    왜 관리자와 기여자가 changelog를 신경써야하는지, 또한 이 프로젝트의 동기에 대해 이야기하기 위해
    #{link_to "The Changelog 팟캐스트", data.links.thechangelog}에 다녀왔습니다.
```

## File: source/layouts/layout.html.haml
```haml
// Variables
- latest_version = current_page.metadata[:page][:version] == $last_version
- legacy_version = current_page.metadata[:page][:version] == "0.3.0"
- language_code = current_page.metadata[:page][:language]
- versions = Dir.entries("source/#{language_code}") - %w[. ..]
- current_version = current_page.metadata[:page][:version]
- newer_version_available = File.exist?("source/#{language_code}/#{$last_version}")
- description = current_page.data.description || "bullshit" 

!!! 5
%html{ lang: current_page.data.language }
  %head
    %meta{ charset: 'utf-8' }
    %meta{ content: 'IE=edge', 'http-equiv' => 'X-UA-Compatible' }
    %meta{ name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
    %meta{ name: 'description', content: current_page.data.description }

    -# Open Graph

    %meta{ property: 'og:article:publisher', content: config.publisher_url }
    %meta{ property: 'og:title', content: current_page.data.title }
    %meta{ property: 'og:type', content: 'article' }
    %meta{ property: 'og:url', content: path_to_url(current_page.url) }
    %meta{ property: 'og:description', content: description }
    %meta{ property: 'og:image', content: path_to_url(image_path("keep-a-changelog-opengraph.png")) }
    = yield_content :og

    -# Icons

    %link{ rel: "shortcut icon", type: "image/x-icon", href: image_path("favicon.ico") }
    %link{ rel: 'canonical', href: path_to_url(current_page.url) }

    %link{ rel: "me", href: "https://ruby.social/@olivierlacan" }

    %title= current_page.data.title

    %link{ rel: "preconnect", href: "https://fonts.googleapis.com" }
    %link{ rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: true }

    = stylesheet_link_tag "//fonts.googleapis.com/css?family=Muli:400,700&display=swap"
    = stylesheet_link_tag "//fonts.googleapis.com/css?family=Source+Code+Pro:400,700&display=swap"
    - if legacy_version
      = stylesheet_link_tag 'legacy'
    - else
      = stylesheet_link_tag 'application'
    = javascript_include_tag 'all', defer: true

  %body
    %article
      %header
        - if !latest_version
          - if versions.include?($last_version)
            .newer
              - if $languages[language_code][:new]
                = "#{$languages[language_code][:new]}: "
              - else
                There is a newer version available:
              = link_to "#{$languages[language_code][:name]} #{$last_version}", "/#{language_code}/#{$last_version}"
          - else
            - if $languages[language_code][:notice]
              .last-version-notice= $languages[language_code][:notice]
            - else
              .last-version-notice
                The latest version (#{$last_version}) is not yet available in
                this language but
                = link_to "you can read it in English", "/en/#{$last_version}"
                for now and
                = link_to "help translate ", "https://github.com/olivierlacan/keep-a-changelog/issues"
                it.

        %nav.locales{ role: "navigation" }
          %label{ for: "language-select", title: "Pick one of the #{$language_count} translations" }= "Languages (#{$languages.count}):"
          %select{ name: "language", id: "language-select" }
            - $languages.each do |language|
              - selected = language_code == language.first
              - if available_translation = available_translation_for(language)
                %option{ selected: selected, label: available_translation, value: language.first }
                  = available_translation
        - if !legacy_version
          = image_tag "keep-a-changelog-mark.svg", width: 130, height: 100, class: "mark"

      .main{ role: "main" }
        = yield

        %footer.footer.clearfix{ role: "contentinfo" }
          = image_tag "keep-a-changelog-mark.svg", width: 30, height: 30, class: "mark"

          %p.about
            This project is
            = link_to "MIT Licensed", "https://choosealicense.com/licenses/mit/"
            \ //
            = link_to "Created & maintained", "https://github.com/olivierlacan/keep-a-changelog/"
            by
            = link_to "Olivier Lacan", "https://olivierlacan.com/"
            \ //
            Designed by Tyler Fortune

  - unless config.gauges_id.blank?
    :javascript
      var _gauges = _gauges || [];
      (function() {
        var t   = document.createElement('script');
        t.type  = 'text/javascript';
        t.async = true;
        t.id    = 'gauges-tracker';
        t.setAttribute('data-site-id', '#{config.gauges_id}');
        t.src = '//secure.gaug.es/track.js';
        var s = document.getElementsByTagName('script')[0];
        s.parentNode.insertBefore(t, s);
      })();
```

## File: source/nb/1.1.0/index.html.haml
```haml
---
title: Lag en Endringslogg
description: Ikke la vennene dine dumpe git logger i endringslogger.
language: nb
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versjon
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Hva er en endringslogg?

  %p
    En endringslogg er en fil som inneholder en organisert, kronologisk
    satt opp liste over sentrale endringer for hver versjon av et prosjekt.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Hvorfor lage en endringslogg?

  %p
    For å gjøre det enklere for brukere og bidragsytere å se nøyaktig hvilke
    sentrale endringer som har blitt gjort mellom hver utgivelse (eller versjon)
    av prosjektet.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Hvem trenger en endringslogg?

  %p
    Folk. Om enn de er konsumenter eller utviklere, sluttbrukerne
    av programvare er mennesker som bryr seg om hva som er i programvaren.
    Når programvaren endres, vil folk vite hvorfor og hvordan.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Hvordan lager jeg en god endringslogg?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Veiledende prinsipper

  %ul
    %li
      Endringslogger er <em>for mennesker</em>, ikke maskiner.
    %li
      Det bør være en oppføring for hver versjon.
    %li
      Samme type endringer bør grupperes.
    %li
      Versjoner og seksjoner bør kunne lenkes til.
    %li
      Den nyeste versjonen bør komme først.
    %li
      Utgivelsesdatoen for hver versjon vises.
    %li
      Nevn hvorvidt du følger #{link_to "Semantisk Versjonering", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Type endringer

  %ul
    %li
      %code Lagt til
      for ny funksjonalitet.
    %li
      %code Endret
      for endringer i eksisterende funksjonalitet.
    %li
      %code Avviklet
      for funksjonalitet som snart fjernes.
    %li
      %code Fjernet
      for fjernet funksjonalitet.
    %li
      %code Fikset
      for feilrettinger.
    %li
      %code Sikkerhet
      i tilfelle sårbarheter.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Hvorfor kan jeg redusere innsatsen som må til for å vedlikeholde en endringslogg?

  %p
    Bruk en <code>Ikke utgitt</code>-seksjon øverst for å spore
    kommende endringer.

  %p Dette tjener to formål:

  %ul
    %li
      Folk kan se hvilke endringer de kan forvente i kommende utgivelser
    %li
      Når utgivelsen publiseres kan du flytte <code>Ikke utgitt</code>-seksjonen
      til en seksjon for en ny versjon.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kan endringslogger være dårlige?

  %p Ja. Her er noen måter de kan være mindre nyttige.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Handlingslogg med endringer

  %p
    Å bruke endringer i handlingslogg (<em>commit log diffs</em>)
    som endringslogg er en dårlig idé: De er full av støy. Ting som
    sammenslåing av endringer, obskure titler, endringer i dokumentasjon,
    osv.

  %p
    Formålet med en handlingslogg er å dokumentere et steg i utviklingen
    av kildekoden. Noen prosjekter rydder handlingslogger, andre ikke.

  %p
    Formålet med en oppføring i endringslogg er å dokumentere
    sentrale endringer, gjerne på tvers av flere handlingslogger,
    samt å kommunisere disse klart til sluttbrukerne.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorere Avviklinger

  %p
    Når folk oppgraderer fra en versjon til en annen burde det
    være smertelig tydelig når noe vil gå i stykker. Det burde være
    mulig å oppgradere til en versjon som oppgir avviklinger, fjerne
    det som er avviklet, og deretter oppgradere til en versjon hvor
    avviklinger blir det som er fjernet.

  %p
    Om ikke annet, oppgi avviklinger, hva som er fjernet, og annet
    som gjør at noe går i stykker i endringsloggen.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Forvirrende Datoer

  %p
    Regionale datoformater varierer, og det er ofte vanskelig å
    finne et menneskevennlig datoformat som føles intuitivt for alle.
    Fordelen med datoformater som <code>2017-07-17</code> er at de
    følger rekkefølgen med største til minste enhet: År, måned og dato.
    Dette formatet overlapper heller ikke på tvetydige måter, til
    forskjell fra noen regionale formater som bytter posisjonen til
    måneds- og datonumre. Derfor, og fordi dette datoformatet er en
    #{link_to "ISO-standard", data.links.isodate}, er det anbefalt datoformat for
    oppføringer i endringslogger.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Tvetydige Endringer

  %p
    En endringslogg som bare nevner noen av endringene kan være like
    farlig som å ikke ha en endringslogg. Selv om mange av endringene
    ikke er relevante - for eksempel, fjerning av et enkelt mellomrom
    ikke trenger å registreres i alle tilfeller - bør alle sentrale
    endringer nevnes i endringsloggen. Ved inkonsistent oppførsel av
    endringer vil brukerne dine feilaktig tro at endringsloggen er
    den eneste kilden til sannhet. Den bør være det. Ved stor makt
    kommer stort ansvar - å ha en god endringslogg betyr å ha en
    konsistent oppdatert endringslogg.

  %aside
    Det er mer. Hjelp meg med å samle disse antimønstrene ved å
    = link_to "åpne en sak", data.links.issue
    saker eller et endringsforslag.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Ofte Spurte Spørsmål

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Er det et standard format for endringslogger?

  %p
    Ikke egentlig. Det finnes #{link_to "GNUs stilguide for endringslogger", data.links.gnustyle},
    eller den #{link_to "to avsnitt lange GNU NEWS filen", data.links.gnunews}
    "retningslinjen". Begge er inadekvate eller utilstrekkelige.

  %p
    Dette prosjektet søker å være 
    = link_to "en bedre konvensjon for endringslogger", data.links.changelog
    Dette kommer fra observasjon av beste praksis i miljøet
    for åpen kildekode og innsamling av de.

  %p
    Sunn kritikk, diskusjon og forslag til forbedringer
    = link_to "er velkomne.", data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Hvordan bør endringsloggfilen navngis?

  %p
    Kall de <code>CHANGELOG.md</code>. Noen prosjekter bruker
    <code>HISTORY</code>, <code>NEWS</code> eller <code>RELEASES</code>.

  %p
    Selv om det er enkelt å tenke at navnet på endringsloggfilen
    ikke betyr så mye, hvorfor gjøre det vanskeligere for sluttbrukerne
    å konsekvent finne sentrale endringer?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Hva med utgivelser på GitHub?

  %p
    Det er et flott initiativ. #{link_to "Utgivelser", data.links.github_releases} kan brukes
    for å gjøre enkle taggede versjoner på git (for eksempel <code>v1.0.0</code>)
    om til fyldige utgivelsesnotater ved å manuelt legge de til, eller
    å hente annoterte notater fra taggede versjoner å gjøre de om til
    utgivelsesnotater.

  %p
    Utgivelser på GitHub lager en ikke-portabel endringslogg som bare
    kan vises til brukerne innenfor konteksten GitHub. Det er mulig å
    gjøre de veldig lik formatet til Lag en Endringslogg, men det er
    vanligvis mer krevende.

  %p
    Den nåværende versjonen av utgivelser på GitHub er antageligvis
    ikke veldig gjenfinnbar for sluttbrukere, i motsetning til de
    typiske filene med store bokstaver (<code>README</code>, 
    <code>CONTRIBUTING</code>, osv.) Et annet, mindre, problem er at
    grensesnittet per nå ikke tilbyr lenker til handlingslogger
    mellom hver utgivelse.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Kan endringslogger automatisk tolkes?

  %p
    Det er vanskelig, fordi folk følger svært forskjellige formater
    og filnavn.

  %p
    #{link_to "Vandamme", data.links.vandamme} er en Ruby gem laget av Gemnasium-teamet
    for å tolke mange (men ikke alle) endringslogger for prosjekter med
    åpen kildekode.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Hva med tilbaketrukne utgivelser?

  %p
    Tilbaketrukne utgivelser er versjoner som måtte trekkes tilbake
    på grunn av en alvorlig feil eller et sikkerhetsproblem. Vanligvis
    fremgår ikke disse versjonene i endringsloggen. Det bør det. Slik
    bør de vises frem:

  %p <code>## [0.0.5] - 2014-12-13 [TILBAKETRUKKET]</code>

  %p
    Merkelappen <code>[TILBAKETRUKKET]</code> er bevisst uthevet. Det
    er for at folk skal legge merke til den. Siden den er omgitt av
    braketter er den også enklere å tolke programmatisk.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Bør du noensinne omskrive en endringslogg?

  %p
    Selvfølgelig. Det er alltid gode grunner til å forbedre en
    endringslogg. Jeg lager regelmessig endringsforslag for å legge
    til manglende utgivelser i prosjekter med åpen kildekode som
    mangler vedlikeholdte endringslogger.

  %p
    Det er også mulig at du oppdager at du glemte å addressere en
    sentral endring i notatene til en versjon. Det er selvfølgelig
    viktig at du oppdaterer endringsloggen i dette tilfellet.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Hvordan kan jeg bidra?

  %p
    Dette dokumentet er ikke <strong>sannheten</strong>; det er min
    nøye vurderte mening, samt informasjon og eksempler jeg har samlet.

  %p
    Dette fordi jeg vil at vårt miljø skal nå en enighet. Jeg tror
    at diskusjonen er like viktig som resultatet.

  %p
    Så vær så snill, <strong>#{link_to "bidra", data.links.repo}</strong>.

.press
  %h3 Samtaler
  %p
    Jeg deltok på #{link_to "The Changelog podcast", data.links.thechangelog} for
    å snakke om hvorfor vedlikeholdere og bidragsytere burde bry seg om
    endringslogger, og også om motivasjonene bak dette prosjektet.
```

## File: source/nl/1.0.0/index.html.haml
```haml
---
description: Laat je vrienden geen git logs in changelogs dumpen.
title: Keep a Changelog
language: nl
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versie
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Wat is een changelog?

  %p
    Een changelog is een bestand met een zorgvuldig samengestelde, chronologische lijst
    van noemenswaardige aanpassingen voor elke versie van een project.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Waarom een changelog bijhouden?

  %p
    Om het makkelijker te maken voor gebruikers en programmeurs om precies te zien welke
    noemenswaardige aanpassingen er gedaan zijn tussen elke release (of versie) van het project.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Wie heeft een changelog nodig?

  %p
    Mensen hebben het nodig. Of het nu consumenten of ontwikkelaars zijn, eindgebruikers van
    software zijn mensen die er om geven wat er in de software zit die ze gebruiken.
    Als de software verandert, wil men weten wat en hoe.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Hoe maak ik een goed changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Richtlijnen

  %ul
    %li
      Changelogs zijn <em>voor mensen</em>, niet voor machines.
    %li
      Er zou een vermelding moeten zijn voor elke versie.
    %li
      Aanpassingen van het zelfde type moeten gegroepeerd worden.
    %li
      Versies en secties zouden linkbaar moeten zijn.
    %li
      De laatste versie staat bovenaan.
    %li
      De release datum van elke versie word weergegeven.
    %li
      Geef aan of je #{link_to "Semantic Versioning", data.links.semver} gebruikt.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Types of changes

  %ul
    %li
      %code Added
      voor nieuwe functionaliteit.
    %li
      %code Changed
      voor aanpassingen aan bestaande functionaliteit.
    %li
      %code Deprecated
      voor functionaliteit die binnenkort komt te vervallen.
    %li
      %code Removed
      voor functionaliteit die vanaf nu vervallen is.
    %li
      %code Fixed
      voor bug fixes.
    %li
      %code Security
      voor aanpassingen met betrekking tot veiligheid.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Hoe kan ik, met zo min mogelijk moeite, een changelog bij houden?

  %p
    Houd bovenin een <code>Unreleased</code> sectie bij met aanpassingen voor de komende release.

  %p Dit heeft twee doelen:

  %ul
    %li
      Mensen kunnen zien wat te verwachten in de aankomende release.
    %li
      Als je een release doet kan je eenvoudig de <code>Unreleased</code> sectie
      aanpassen naar een nieuwe release sectie.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kan een changelog slecht zijn?

  %p Ja. Hier een paar manieren waarop je een changelog behoorlijk onbruikbaar kan maken.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    Commit log diffs gebruiken als een changelog is een slecht idee:
    ze staan vol met ruis. Denk bijvoorbeeld aan merge commits, commits met
    onduidelijke titels, documentatie aanpassingen etc.

  %p
    Het doel van een commit bericht is om één enkele stap in de evolutie van de
    code te beschrijven.

  %p
    Het doel van een changelog is om noemenswaardige aanpassingen te documenteren,
    vaak over meerdere commits, en om deze duidelijk naar de eindgebruiker te communiceren.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Deprecations negeren

  %p
    Wanneer mensen upgraden van de ene naar de andere versie,
    moet het overduidelijk zijn als er iets niet meer zal werken.
    Het moet mogelijk zijn om te upgraden naar een versie met deprecations,
    vervolgens de deprecations weg te halen, en vervolgens
    de upgrade kunnen doen naar de versie waar de deprecations removals zijn geworden.

  %p
    Geef altijd op zijn minst de deprecations, removals en changes met grote impact aan in je changelog.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Verwarrende datums

  %p
    Datum notaties verschillen van land tot land, en het is vaak moeilijk om
    een notatie te vinden die makkelijk te lezen is en intuïtief is voor iedereen.

    Het voordeel van de notatie <code>2017-07-17</code> is dat het jaar, maand en dag
    op volgorde van grootte laat zien. Daarom, en het feit dat dit een  #{link_to "ISO standaard", data.links.isodate}
    is, is dit de aanbevolen datum notatie voor changelog releases.

  %aside
    Dit is niet alles. Help mij antipatterns te verzamelen door
    = link_to "een issue", data.links.issue
    of een pull request aan te maken.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Veel Gestelde Vragen

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Is er een standaard changelog template?

  %p
    Niet echt. Er is de GNU changelog style guide, en het twee paragrafen lange GNU NEWS bestand "richtlijnen".
    Beiden zijn niet volledig genoeg.

  %p
    Dit project poogt
    = link_to "een betere changelog standaard", data.links.changelog
    te creëren. Dit op basis van "good practices" uit de open source wereld.

  %p
    Opbouwende kritiek, discussie en suggesties voor verbetering
    = link_to "zijn welkom.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Wat zou de changelog bestandsnaam moeten zijn?

  %p
    Noem het <code>CHANGELOG.md</code>. Sommige projecten gebruiken
    <code>HISTORY</code>, <code>NEWS</code> of <code>RELEASES</code>.

  %p
    Je kan denken dat de bestandsnaam niet heel belangrijk is,
    maar waarom zou je het de eindgebruikers moeilijker maken om de changelog te vinden?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Wat denk je van GitHub Releases?

  %p
    Het is een goed initiatief. #{link_to "Releases", data.links.github_releases} kan gebruikt worden
    om simpele git tags (bijvoorbeeld een tag met de naam  <code>v1.0.0</code>)
    te veranderen in uitgebreide release notes door deze handmatig toe te voegen of
    door geannoteerde git tag berichten te gebruiken om release notes te genereren.

  %p
    GitHub Releases maken changelog wat alleen getoond kan worden aan gebruikers
    binnen de context van GitHub. Het is mogelijk om deze dicht bij het format
    te krijgen wat wij hier promoten, maar er zal iets meer werk voor nodig zijn.

  %p
    De huidige versie van GitHub releases is naar mijn mening niet
    echt goed vindbaar voor gebruikers, in tegenstelling tot de typische
    bestanden die in een naam in hoofdletters hebben
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.).
    Een ander knelpunt is dat de interface geen links toestaat naar
    commit logs van elke release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Kunnen changelogs automatisch geparsed worden?

  %p
    Dat is lastig, mensen gebruiken immers veel verschillende formats en bestandsnamen.

  %p
    #{link_to "Vandamme", data.links.vandamme} is een Ruby gem van het
    Gemnasium team wat de changelogs van veel (maar niet alle)
    open source projecten kan parsen.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Wat doen we met teruggetrokken (yanked) releases?

  %p
    Teruggetrokken releases zijn versies die teruggetrokken zijn als gevolg
    van een serieuze bug of beveiligings probleem. Vaak zijn ze niet eens te zien in
    de changelogs. Dat zou wel moeten. Zo zou je een teruggetrokken release moeten tonen:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    De <code>[YANKED]</code> tag is in hoofdletters voor een reden. Het is belangrijk
    dat mensen dit zien. Omdat het tussen blokhaken genoteerd is, is het ook makkelijker
    automatisch te parsen.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Mag je een changelog aanpassen/herschrijven?

  %p
    Natuurlijk. Er zijn goede redenen om een changelog te verbeteren.
    Ik open regelmatig een pull request om missende releases toe te
    voegen aan open source projecten met een slecht onderhouden changelog.

  %p
    Het kan ook zo zijn dat je ontdekt dat je een belangrijke aanpassing niet
    vermeld hebt in je changelog. Het is dan natuurlijk zaak om dit alsnog
    in je changelog te vermelden.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Hoe kan ik bijdragen?

  %p
    Dit document is niet de <strong>waarheid</strong>; het is mijn
    weloverwogen mening, samen met wat informatie en voorbeelden die ik verzameld heb.

  %p
    Dit heb ik gedaan omdat ik wil dat de programmeer gemeenschap een consensus bereikt.
    Ik denk dat de discussie net zo belangrijk is als het eindresultaat.

  %p
    Dus <strong>#{link_to "alle hulp is welkom", data.links.repo}</strong>.

.press
  %h3 Conversaties
  %p
    Ik was te gast bij #{link_to "The Changelog podcast", data.links.thechangelog} om te praten over
    waarom een changelog belangrijk is programmeurs, en over mijn motivatie achter dit project.
```

## File: source/nl/1.1.0/index.html.haml
```haml
---
description: Laat je vrienden geen git logs in changelogs dumpen.
title: Keep a Changelog
language: nl
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versie
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Wat is een changelog?

  %p
    Een changelog is een bestand met een zorgvuldig samengestelde, chronologische lijst
    van noemenswaardige aanpassingen voor elke versie van een project.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Waarom een changelog bijhouden?

  %p
    Om het makkelijker te maken voor gebruikers en programmeurs om precies te zien welke
    noemenswaardige aanpassingen er gedaan zijn tussen elke release (of versie) van het project.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Wie heeft een changelog nodig?

  %p
    Mensen hebben het nodig. Of het nu consumenten of ontwikkelaars zijn, eindgebruikers van
    software zijn mensen die er om geven wat er in de software zit die ze gebruiken.
    Als de software verandert, wil men weten wat en hoe.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Hoe maak ik een goed changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Richtlijnen

  %ul
    %li
      Changelogs zijn <em>voor mensen</em>, niet voor machines.
    %li
      Er zou een vermelding moeten zijn voor elke versie.
    %li
      Aanpassingen van het zelfde type moeten gegroepeerd worden.
    %li
      Versies en secties zouden linkbaar moeten zijn.
    %li
      De laatste versie staat bovenaan.
    %li
      De release datum van elke versie word weergegeven.
    %li
      Geef aan of je #{link_to "Semantic Versioning", data.links.semver} gebruikt.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Types of changes

  %ul
    %li
      %code Added
      voor nieuwe functionaliteit.
    %li
      %code Changed
      voor aanpassingen aan bestaande functionaliteit.
    %li
      %code Deprecated
      voor functionaliteit die binnenkort komt te vervallen.
    %li
      %code Removed
      voor functionaliteit die vanaf nu vervallen is.
    %li
      %code Fixed
      voor bug fixes.
    %li
      %code Security
      voor aanpassingen met betrekking tot veiligheid.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Hoe kan ik, met zo min mogelijk moeite, een changelog bij houden?

  %p
    Houd bovenin een <code>Unreleased</code> sectie bij met aanpassingen voor de komende release.

  %p Dit heeft twee doelen:

  %ul
    %li
      Mensen kunnen zien wat te verwachten in de aankomende release.
    %li
      Als je een release doet kan je eenvoudig de <code>Unreleased</code> sectie
      aanpassen naar een nieuwe release sectie.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kan een changelog slecht zijn?

  %p Ja. Hier een paar manieren waarop je een changelog behoorlijk onbruikbaar kan maken.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffs

  %p
    Commit log diffs gebruiken als een changelog is een slecht idee:
    ze staan vol met ruis. Denk bijvoorbeeld aan merge commits, commits met
    onduidelijke titels, documentatie aanpassingen etc.

  %p
    Het doel van een commit bericht is om één enkele stap in de evolutie van de
    code te beschrijven.

  %p
    Het doel van een changelog is om noemenswaardige aanpassingen te documenteren,
    vaak over meerdere commits, en om deze duidelijk naar de eindgebruiker te communiceren.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Deprecations negeren

  %p
    Wanneer mensen upgraden van de ene naar de andere versie,
    moet het overduidelijk zijn als er iets niet meer zal werken.
    Het moet mogelijk zijn om te upgraden naar een versie met deprecations,
    vervolgens de deprecations weg te halen, en vervolgens
    de upgrade kunnen doen naar de versie waar de deprecations removals zijn geworden.

  %p
    Geef altijd op zijn minst de deprecations, removals en changes met grote impact aan in je changelog.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Verwarrende datums

  %p
    Datum notaties verschillen van land tot land, en het is vaak moeilijk om
    een notatie te vinden die makkelijk te lezen is en intuïtief is voor iedereen.

    Het voordeel van de notatie <code>2017-07-17</code> is dat het jaar, maand en dag
    op volgorde van grootte laat zien. Daarom, en het feit dat dit een  #{link_to "ISO standaard", data.links.isodate}
    is, is dit de aanbevolen datum notatie voor changelog releases.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Inconsistente Aanpassingen

  %p
    Een changelog waar slechts een aantal van de aanpassingen in vermeld staan,
    kan net zo gevaarlijk zijn als geen changelog hebben.
    Sommige aanpassingen zullen te irrelevant zijn - bijvoorbeeld, het weghalen
    van een overbodige spatie hoeft niet altijd vermeld te worden - maar alle
    belangrijke aanpassingen zouden in het changelog vermeld moeten worden.
    Door inconsistent te loggen kunnen gebruikers onterecht denken dat de changelog
    de complete bron van waarheid is. Dit zou wel zo moeten zijn.
    "With great power comes great responsibility". Als je een changelog bijhoudt,
    zorg er dan voor dat deze continue bijgehouden word.

  %aside
    Dit is niet alles. Help mij antipatterns te verzamelen door
    = link_to "een issue", data.links.issue
    of een pull request aan te maken.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Veel Gestelde Vragen

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Is er een standaard changelog template?

  %p
    Niet echt. Er is de GNU changelog style guide, en het twee paragrafen lange GNU NEWS bestand "richtlijnen".
    Beiden zijn niet volledig genoeg.

  %p
    Dit project poogt
    = link_to "een betere changelog standaard", data.links.changelog
    te creëren. Dit op basis van "good practices" uit de open source wereld.

  %p
    Opbouwende kritiek, discussie en suggesties voor verbetering
    = link_to "zijn welkom.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Wat zou de changelog bestandsnaam moeten zijn?

  %p
    Noem het <code>CHANGELOG.md</code>. Sommige projecten gebruiken
    <code>HISTORY</code>, <code>NEWS</code> of <code>RELEASES</code>.

  %p
    Je kan denken dat de bestandsnaam niet heel belangrijk is,
    maar waarom zou je het de eindgebruikers moeilijker maken om de changelog te vinden?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Wat denk je van GitHub Releases?

  %p
    Het is een goed initiatief. #{link_to "Releases", data.links.github_releases} kan gebruikt worden
    om simpele git tags (bijvoorbeeld een tag met de naam  <code>v1.0.0</code>)
    te veranderen in uitgebreide release notes door deze handmatig toe te voegen of
    door geannoteerde git tag berichten te gebruiken om release notes te genereren.

  %p
    GitHub Releases maken changelog wat alleen getoond kan worden aan gebruikers
    binnen de context van GitHub. Het is mogelijk om deze dicht bij het format
    te krijgen wat wij hier promoten, maar er zal iets meer werk voor nodig zijn.

  %p
    De huidige versie van GitHub releases is naar mijn mening niet
    echt goed vindbaar voor gebruikers, in tegenstelling tot de typische
    bestanden die in een naam in hoofdletters hebben
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.).
    Een ander knelpunt is dat de interface geen links toestaat naar
    commit logs van elke release.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Kunnen changelogs automatisch geparsed worden?

  %p
    Dat is lastig, mensen gebruiken immers veel verschillende formats en bestandsnamen.

  %p
    #{link_to "Vandamme", data.links.vandamme} is een Ruby gem van het
    Gemnasium team wat de changelogs van veel (maar niet alle)
    open source projecten kan parsen.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Wat doen we met teruggetrokken (yanked) releases?

  %p
    Teruggetrokken releases zijn versies die teruggetrokken zijn als gevolg
    van een serieuze bug of beveiligings probleem. Vaak zijn ze niet eens te zien in
    de changelogs. Dat zou wel moeten. Zo zou je een teruggetrokken release moeten tonen:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    De <code>[YANKED]</code> tag is in hoofdletters voor een reden. Het is belangrijk
    dat mensen dit zien. Omdat het tussen blokhaken genoteerd is, is het ook makkelijker
    automatisch te parsen.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Mag je een changelog aanpassen/herschrijven?

  %p
    Natuurlijk. Er zijn goede redenen om een changelog te verbeteren.
    Ik open regelmatig een pull request om missende releases toe te
    voegen aan open source projecten met een slecht onderhouden changelog.

  %p
    Het kan ook zo zijn dat je ontdekt dat je een belangrijke aanpassing niet
    vermeld hebt in je changelog. Het is dan natuurlijk zaak om dit alsnog
    in je changelog te vermelden.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Hoe kan ik bijdragen?

  %p
    Dit document is niet de <strong>waarheid</strong>; het is mijn
    weloverwogen mening, samen met wat informatie en voorbeelden die ik verzameld heb.

  %p
    Dit heb ik gedaan omdat ik wil dat de programmeer gemeenschap een consensus bereikt.
    Ik denk dat de discussie net zo belangrijk is als het eindresultaat.

  %p
    Dus <strong>#{link_to "alle hulp is welkom", data.links.repo}</strong>.

.press
  %h3 Conversaties
  %p
    Ik was te gast bij #{link_to "The Changelog podcast", data.links.thechangelog} om te praten over
    waarom een changelog belangrijk is programmeurs, en over mijn motivatie achter dit project.
```

## File: source/pl/0.3.0/index.html.haml
```haml
---
title: Prowadź Changelog
description: Nie pozwól, by twoi znajomi wklejali logi gita do CHANGELOGów
language: pl
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Czym jest changelog?
  Changelog, inaczej rejestr zmian lub historia zmian, to plik zawierający chronologicznie uporządkowaną listę istotnych zmian dla każdej wersji projektu.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Co jest celem prowadzenia changelogu?
  Changelog pomaga użytkownikom zrozumieć, jakie znaczące zmiany zostały wprowadzone w każdej wersji projektu.

  ### Dlaczego miałoby mi zależeć?
  Ponieważ narzędzia informatyczne są dla ludzi. Jeśli ci nie zależy,
  dlaczego przyczyniasz się do powstawania otwartego oprogramowania (open source)?

  Rozmawiałem na podcaście ["The Changelog" z Adamem Stacoviakiem i Jerodem Santo][thechangelog]
  (odpowiednia nazwa, prawda?) o tym, dlaczego osobom wspierającym otwarte programowanie powinno zależeć,
  oraz o celach samego projektu. Jeśli masz chwilę (1:06), zapraszam do posłuchania - warto.

  ### Co składa się na dobry changelog?
  Cieszę się, że zapytałeś.

  Dobry changelog trzyma się następujących zasad:

  - Jest stworzony dla ludzi, nie maszyn, więc liczy się czytelność.
  - Prostota dodawania linków do każdego rozdziału (dlatego używa się Markdown zamiast prostego tekstu).
  - Jeden podrozdział dla każdej wersji.
  - Wyszczególniaj wydania w odwrotnym porządku chronologicznym (najnowsza na górze).
  - Wszystkie daty zapisuj w formacie `YYYY-MM-DD`. (Przykład: `2012-06-02` dla `2 czerwca 2012 r.`). To [rozsądny](https://xkcd.com/1179/), niezależny od języka międzynarodowy format.
  - Zawsze określaj, czy projekt jest zgodny z [Semantycznym Wersjonowaniem][semver].
  - Każda wersja powinna:
    - Zawierać datę w wyżej wymienionym formacie.
    - Grupować zmiany w celu opisu ich wpływu na projekt, w następujący sposób:
      - `Added` dla nowych funkcji.
      - `Changed` dla zmian w istniejącej funkcjonalności.
      - `Deprecated` (przestarzałe) dla uprzednio stabilnych funkcji, które zostaną usunięte w przyszłych wydaniach projektu.
      - `Removed` dla przestarzałych funkcji usuniętych w bieżącej wersji.
      - `Fixed` dla poprawionych błędów (bugów).
      - `Security` w celu poinformowania użytkowników o zalecanej aktualizacji naprawiającej luki bezpieczeństwa.

  ### Jak mogę zminimalizować wysiłek wkładany w prowadzenie changelogu?
  Zawsze umieszczaj rozdział `"Unreleased"` na szczycie dokumentu, w celu śledzenia wszelkich zmian.

  Ta praktyka ma dwa cele:

  - Użytkownicy widzą, jakich zmian mogą się spodziewać w nadchodzących wydaniach.
  - W momencie aktualizacji, musisz jedynie zastąpić `"Unreleased"` wersją projektu oraz dodać nowy nagłówek `"Unreleased"` na samej górze.

  ### Co sprawia, że jednorożce płaczą?
  Dobrze... zabierzmy się za to.

  - **Wrzucanie logów diff z zamieszczonymi zmianami.** Po prostu tego nie rób. Nikomu tym nie pomagasz.
  - **Niewyszczególnianie przestarzałych funkcji.** Użytkownik powinien zostać wyraźnie poinformowany, że coś przestanie działać po aktualizacji.
  - **Daty w formatach regionalnych.** W USA ludzie zapisują miesiąc na samym początku daty ("06-02-2012" dla 2 czerwca 2012 r.,
    co nie ma *najmniejszego* sensu), podczas gdy gdzie indziej wiele osób pisze "2 czerwiec 2012", jednak wymawia to w inny sposób.
    "2012-06-02" to logiczny format zapisywany od największej, do najmniejszej wartości oraz nie pokrywa się z innymi formatami w niejednoznaczny sposób.
    Jest to jednocześnie [standard ISO](http://www.iso.org/iso/home/standards/iso8601.htm). To wszystko sprawia, że jest to rekomendowany format zapisywania daty w changelogach.

  To jeszcze nie koniec. Pomóż mi zebrać wszystkie łzy jednorożców [zgłaszając problem][issues] lub wprowadzając zmianę poprzez pull request.

  ### Czy istnieje standardowy format changelogu?
  Niestety nie, ale spokojnie. Wiem, że spieszysz wkleić link do tego przewodnika stylu changelogu GNU, czy do dwóch paragrafów "wytycznych" GNU NEWS.
  Przewodnik stylu GNU to dobry, ale niestety naiwny początek. Nie ma nic złego w byciu naiwnym, ale gdy ludzie potrzebują wytycznych, bycie naiwnym rzadko pomaga.
  Szczególnie, gdy istnieje wiele sytuacji i szczególnych przypadków z którymi trzeba się zmierzyć.

  Mam nadzieję, że ten projekt zostanie uznany [lepszym standardem pliku CHANGELOG][CHANGELOG].

  Nie uważam, że status quo jest wystarczające i myślę, że jako społeczność, możemy stworzyć lepsze konwencje,
  jeśli spróbujemy zastosować dobre praktyki stosowane w prawdziwych projektach informatycznych.
  Proszę, zapoznaj się z projektem i pamiętaj, że [dyskusja i sugestie są zawsze mile widziane][issues]!

  ### Jak powinien być nazwany plik changelog?
  Jeśli nie potrafisz wywnioskować z powyższego przykładu, `CHANGELOG.md` to do tej pory najlepsza konwencja.

  Niektóre projekty również stosują `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`, itd.

  Straszny bałagan. Wszystkie te nazwy sprawiają, że jeszcze ciężej jest odnaleźć ten plik.

  ### Dlaczego nie mielibyśmy po prostu stosować raportu `git log`?
  Ponieważ logi typu diff są z natury zagmatwane. Nawet przy hipotetycznym projekcie stworzonym przez doskonałe istoty ludzkie, które nigdy nie popełniają literówek,
  nigdy nie zapominają o zsynchronizowaniu nowo dodanych plików czy nigdy niczego nie pomijają podczas refaktoryzacji kodu, logi diff nie mogłyby zastąpić changelogu.
  Celem `git commit` jest udokumentowanie jednego kroku w procesie, dzięki któremu kod ewoluuje z jednego stanu w kolejny. Celem changelogu jest udokumentowanie tych różnic pomiędzy stanami kodu, które są godne uwagi.

  Różnica między changelogiem a logiem diff, tak samo jak różnica między dobrymi komentarzami a kodem, jest następująca: pierwsze opisuje *dlaczego*, drugie - *jak*.

  ### Czy changelog może być generowany automatycznie?
  To nie takie proste, ponieważ ludzie stosują bardzo różne formaty oraz nazwy plików.

  [Vandamme][vandamme] to gem Ruby stworzony przez ekipę [Gemnasium][gemnasium], parsujący changelogi wielu (ale nie wszystkich) projektów open source.

  ### Dlaczego czasem stosujesz pisownię "CHANGELOG", a czasem "changelog"?
  "CHANGELOG" to nazwa samego pliku. Wygląda to dość głośno, ale taka jest historyczna konwencja przyjęta przez wiele projektów open source. Inne przykłady podobnych plików to [`README`][README], [`LICENSE`][LICENSE],
  oraz [`CONTRIBUTING`][CONTRIBUTING].

  Zapis wielkimi literami (dzięki któremu w starszych systemach operacyjnch pliki zostają umieszczone na szczycie listy) ma na celu zwrócenie na nie uwagi.
  Jako że są to ważne informacje dotyczące projektu, mogą być one użyteczne dla każdego, kto zamierza korzystać lub wnieść weń wkład. Ida ta zbliżona jest do [odznaczeń przy projektach open source][shields].

  Gdy stosuję pisownię "changelog", mówię o samej funkcji tego pliku: rejestrowaniu zmian.

  ### A co z błędnymi wersjami (yanked)?
  Są to wersje opublikowane błędnie, czyli takie, które musiały zostać wycofane ze względu na poważny błąd lub lukę bezpieczeństwa. Często takie wersje projektu nie pojawiają się nawet w changelogu, a powinny.
  Oto jak taka wersja powinna wyglądać:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Tag `[YANKED]` jest celowo zapisany wielkimi literami. Ważne jest, by został on dostrzeżony. Dzięki temu, że jest ujęty w nawias, może być z łatwością generowany automatycznie.

  ### Czy powinno się kiedykolwiek poprawiać changelog?
  Oczywiście. Zawsze istnieją powody, by ulepszyć changelog. Często zdarza mi się edytować changelogi w projektach open source w których pominięto udokumentowanie istotnych zmian.

  Może się również zdarzyć, że odkryjesz, iż podczas przygotowywania notatek dla wersji projektu, zapomniałeś udokumentować istotną zmianę, która ma wpływ na działanie aplikacji.
  Ważne jest, by zaktualizować changelog szczególnie jeśli jest to zmiana, która w istotny sposób wpływa na kompatybilność aplikacji.

  ### Jak mogę wnieść wkład w projekt?
  Ten dokument to nie jedna i jedyna **prawda**; to moja starannie sformułowana opinia wraz z informacją i przykładami które zebrałem.
  Pomimo tego, że prowadzę właściwy [CHANGELOG][] na [GitHubie][gh], celowo nie stworzyłem poprawnego *releasu* czy jasno sprecyzowanych wytycznych (tak jak np. na [SemVer.org][semver]).

  Chcę, by nasza społeczność uzgodniła jak CHANGELOG ma wyglądać. Wierzę, że dyskusja jest niezbędna do osiągnięcia końcowego rezultatu.

  Więc proszę, [**dołącz do projektu**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/pl/1.0.0/index.html.haml
```haml
---
title: Prowadź Changelog
description: Nie pozwól swoim znajomym wklejać logów Gita do changelogów.
language: pl
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Wersja
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Czym jest changelog?
  %p
    Changelog, inaczej rejestr zmian, to plik zawierający utrzymywaną,
    uporządkowaną chronologicznie, listę istotnych zmian dla każdej wersji
    projektu.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Po co prowadzić changelog?
  %p
    Aby użytkownikom i deweloperom łatwiej było dokładnie zobaczyć, jakie
    znaczące zmiany zostały wprowadzane w każdym wydaniu (lub wersji) projektu.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Komu potrzebny jest changelog?

  %p
    Ludziom. Czy to klienci czy deweloperzy, końcowi użytkownicy oprogramowania
    są istotami ludzkimi, którym nie jest obojętne, co jest w oprogramowaniu.
    Kiedy oprogramowanie się zmienia, ludzie chcą wiedzieć dlaczego i jak.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Jak zrobić dobry changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Zasady przewodnie

  %ul
    %li
      Changelogi są <em>dla ludzi</em>, nie maszyn.
    %li
      Każda wersja powinna mieć swój wpis.
    %li
      Jednakowe typy zmian powinny być zgrupowane.
    %li
      Wersje i sekcje powinny być linkowalne.
    %li
      Najnowsza wersja na pierwszym miejscu.
    %li
      Wyszczególniona data wydania każdej wersji.
    %li
      Wzmianka, czy przestrzegacie #{link_to "wersjonowania semantycznego", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Typy zmian

  %ul
    %li
      %code Dodane
      dla nowych funkcjonalności.
    %li
      %code Zmienione
      dla zmian w istniejących funkcjonalnościach.
    %li
      %code Zdezaprobowane
      dla funkcjonalności wkrótce do usunięcia.
    %li
      %code Usunięte
      dla teraz usuwanych funkcjonalności.
    %li
      %code Naprawione
      dla jakichkolwiek poprawek błędów.
    %li
      %code Bezpieczeństwo
      w przypadku luk w&nbsp;zabezpieczeniach.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Jak mogę zminimalizować wysiłek wkładany w prowadzenie changelogu?
  %p
    Prowadź sekcję <code>Niewydane</code> na szczycie dokumentu, aby śledzić
    nadchodzące zmiany.

  %p Ta praktyka ma dwa cele:

  %ul
    %li
      Ludzie widzą, jakich zmian mogą się spodziewać w&nbsp;nadchodzących
      wydaniach.
    %li
      W momencie wydania możesz przenieść zmiany z sekcji <code>Niewydane</code>
      do sekcji nowego wydania.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Czy changelogi mogą być złe?

  %p Tak. Oto kilka sposobów, w jakie mogą być mniej niż użyteczne.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Zmiany w commit logu

  %p
    Używanie zmian w commit logach jako changelogów jest złym pomysłem: commit
    logi są pełne szumu: rzeczy takich jak merge commity, commity z niejasnymi
    tytułami, zmiany w&nbsp;dokumentacji itp.

  %p
    Zadaniem commita jest udokumentowanie kroku w&nbsp;ewolucji kodu źródłowego.
    Niektóre projekty porządkują commity, niektóre nie.

  %p
    Zadaniem wpisu w changelogu jest udokumentowanie zmian godnych odnotowania,
    często składających się z&nbsp;wielu commitów, aby przedstawić je jasno
    użytkownikom końcowym.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorowanie dezaprobowań

  %p
    Gdy ludzie robią upgrade z jednej wersji do drugiej, powinno być bezboleśnie
    jasne kiedy coś się może zepsuć. Powinno być możliwe upgrade'owanie się do
    wersji, która wypisuje zdezaprobowania, usunięcie tego, co jest
    zdezaprobowane, następnie upgrade'owanie się do wersji, w której
    dezaprobowane funkcjonalności są usuwane.

  %p
    Jeśli nie robisz nic innego, wypisz w swoim changelogu zdezaprobowania,
    usunięcia i&nbsp;jakiekolwiek zmiany łamiące zgodność wstecz.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Mylące daty

  %p
    Regionalne formaty dat różnią się na świecie i często trudno jest znaleźć
    przyjazny człowiekowi format daty, który wydaje się intuicyjny wszystkim.
    Zaletą dat sformatowanych tak jak <code>2017-07-17</code> jest to, że są one
    uporządkowane od największych do najmniejszych jednostek: roku, miesiąca
    i dnia. Ten format również nie nachodzi w niejednoznaczny sposób na inne
    formaty dat, w przeciwieństwie do niektórych regionalnych formatów, które
    zamieniają miejsce numerów miesiąca i dnia. Z tych powodów i faktu, że
    ten format daty jest #{link_to "standardem ISO", data.links.isodate}, wynika rekomendacja
    tego formatu daty do wpisów w&nbsp;changelogu.

  %aside
    Jest tego więcej. Pomóż mi zebrać te antywzorce
    = link_to "otwierając zgłoszenie", data.links.issue
    lub pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Często zadawane pytania

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Czy istnieje standardowy format changelogu?

  %p
    Niezupełnie. Jest przewodnik stylu changelogu GNU, czy dwuparagrafowe
    „wytyczne” GNU NEWS. Oba dokumenty są nieadekwatne lub niewystarczające.

  %p
    Ten projekt pretenduje do bycia
    = link_to "lepszą konwencją changelogu.", data.links.changelog
    Pochodzi z obserwowania i zebrania dobrych praktyk w społeczności open
    source.

  %p
    Zdrowa krytyka, dyskusja i sugestie poprawek
    = link_to "są mile widziane.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Jak powinien się nazywać plik z changelogiem?

  %p
    Nazwij go <code>CHANGELOG.md</code>. Niektóre projekty używają
    <code>HISTORY</code>, <code>NEWS</code> lub <code>RELEASES</code>.

  %p
    Łatwo jest uznać, że nazwa pliku z changelogiem nie ma większego znaczenia,
    lecz po co utrudniać swoim użytkownikom końcowym odnajdowanie w sposób
    konsekwentny istotnych zmian?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Co z GitHub Releases?

  %p
    To wspaniała inicjatywa. #{link_to "Releases", data.links.github_releases} mogą być używane do
    zmiany prostych tagów Gita (na przykład taga nazwanego <code>v1.0.0</code>)
    w bogate informacje o wydaniach przez ręczne dodanie informacji lub mogą
    wyciągać oznaczone message tagów i przekształcać je w informacje.

  %p
    GitHub Releases tworzą nieprzenośny changelog, który może być prezentowany
    użytkownikom tylko w kontekście GitHuba. Można go bardzo upodobnić do
    formatu Prowadź changelog, ale będzie to dość skomplikowane.

  %p
    Bieżąca wersja wydań GitHub jest też prawdopodobnie nienajłatwiejsza do
    odnalezienia dla użytkowników końcowych, w przeciwieństwie do plików
    o nazwach z wielkimi literami (<code>README</code>, <code>CONTRIBUTING</code>
    itp.). Innym mniejszym brakiem jest to, że interfejs obecnie nie posiada
    linków do logów commitów pomiędzy każdymi wydaniami.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Czy changelogi mogą być parsowane automatycznie?

  %p
    To trudne, ponieważ ludzie stosują bardzo różne formaty i nazwy plików.

  %p
    #{link_to "Vandamme", data.links.vandamme} jest gemem Ruby stworzonym przez zespół
    Gemnasium i który parsuje wiele (ale nie wszystkie)
    changelogów projektów open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Co z wycofanymi wydaniami?

  %p
    Wydania typu yanked to wersje, które musiały zostać usunięte z powodu
    poważnego błędu lub problemów z bezpieczeństwem. Często te wersje nie
    pojawiają się w dziennikach zmian. Powinny. Tak powinieneś je wyświetlać:

  %p <code>## 0.0.5 - 2014-12-13 [WYCOFANA]</code>

  %p
    Etykieta <code>[WYCOFANA]</code> jest celowo zapisana wielkimi literami.
    Ważne jest, by zwracano na nią uwagę. Jest otoczona nawiasami, więc jest
    również prostsza do sparsowania przez skrypt.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Czy powinno się kiedykolwiek przerabiać changelog?

  %p
    Pewnie. Zawsze istnieją dobre powody, by ulepszyć changelog. Regularnie
    otwieram pull requesty dodające brakujące wydania do open-source'owych
    projektów z nieutrzymywanymi changelogami.

  %p
    Może się również zdarzyć, że odkryjesz, iż zapomniałeś udokumentować zmianę
    zrywającą zgodność wsteczną w notatkach dla wersji. Oczywiście ważne jest,
    abyś zaktualizował swój changelog w tym przypadku.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Jak mogę wnieść swój wkład?

  %p
    Ten dokument nie jest obiektywną <strong>prawdą</strong>; jest moją
    starannie przemyślaną opinią, z informacjami i przykładami, które zebrałem.

  %p
    To dlatego, że chcę, aby nasza społeczność osiągnęła konsensus. Wierzę, że
    dyskusja jest tak samo istotna jak efekt końcowy.

  %p
    Więc proszę, <strong>#{link_to "wtrąć się", data.links.repo}</strong>.

.press
  %h3 Rozmowy
  %p
    Byłem na #{link_to "podcaście Changelog", data.links.thechangelog}, aby porozmawiać
    dlaczego opiekunowie i kontrybutorzy powinni dbać o changelogi, a także
    o motywacjach stojących za tym projektem.
```

## File: source/pl/1.1.0/index.html.haml
```haml
---
title: Prowadź Changelog
description: Nie pozwól swoim znajomym wklejać logów Gita do changelogów.
language: pl
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Wersja
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Czym jest changelog?
  %p
    Changelog, inaczej rejestr zmian, to plik zawierający utrzymywaną,
    uporządkowaną chronologicznie, listę istotnych zmian dla każdej wersji
    projektu.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Po co prowadzić changelog?
  %p
    Aby użytkownikom i deweloperom łatwiej było dokładnie zobaczyć, jakie
    znaczące zmiany zostały wprowadzane w każdym wydaniu (lub wersji) projektu.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Komu potrzebny jest changelog?

  %p
    Ludziom. Czy to klienci czy deweloperzy, końcowi użytkownicy oprogramowania
    są istotami ludzkimi, którym nie jest obojętne, co jest w oprogramowaniu.
    Kiedy oprogramowanie się zmienia, ludzie chcą wiedzieć dlaczego i jak.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Jak zrobić dobry changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Zasady przewodnie

  %ul
    %li
      Changelogi są <em>dla ludzi</em>, nie maszyn.
    %li
      Każda wersja powinna mieć swój wpis.
    %li
      Jednakowe typy zmian powinny być zgrupowane.
    %li
      Wersje i sekcje powinny być linkowalne.
    %li
      Najnowsza wersja na pierwszym miejscu.
    %li
      Wyszczególniona data wydania każdej wersji.
    %li
      Wzmianka, czy przestrzegacie #{link_to "wersjonowania semantycznego", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Typy zmian

  %ul
    %li
      %code Dodane
      dla nowych funkcjonalności.
    %li
      %code Zmienione
      dla zmian w istniejących funkcjonalnościach.
    %li
      %code Zdezaprobowane
      dla funkcjonalności wkrótce do usunięcia.
    %li
      %code Usunięte
      dla teraz usuwanych funkcjonalności.
    %li
      %code Naprawione
      dla jakichkolwiek poprawek błędów.
    %li
      %code Bezpieczeństwo
      w przypadku luk w&nbsp;zabezpieczeniach.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Jak mogę zminimalizować wysiłek wkładany w prowadzenie changelogu?
  %p
    Prowadź sekcję <code>Niewydane</code> na szczycie dokumentu, aby śledzić
    nadchodzące zmiany.

  %p Ta praktyka ma dwa cele:

  %ul
    %li
      Ludzie widzą, jakich zmian mogą się spodziewać w&nbsp;nadchodzących
      wydaniach.
    %li
      W momencie wydania możesz przenieść zmiany z sekcji <code>Niewydane</code>
      do sekcji nowego wydania.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Czy changelogi mogą być złe?

  %p Tak. Oto kilka sposobów, w jakie mogą być mniej niż użyteczne.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Zmiany w commit logu

  %p
    Używanie zmian w commit logach jako changelogów jest złym pomysłem: commit
    logi są pełne szumu: rzeczy takich jak merge commity, commity z niejasnymi
    tytułami, zmiany w&nbsp;dokumentacji itp.

  %p
    Zadaniem commita jest udokumentowanie kroku w&nbsp;ewolucji kodu źródłowego.
    Niektóre projekty porządkują commity, niektóre nie.

  %p
    Zadaniem wpisu w changelogu jest udokumentowanie zmian godnych odnotowania,
    często składających się z&nbsp;wielu commitów, aby przedstawić je jasno
    użytkownikom końcowym.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorowanie dezaprobowań

  %p
    Gdy ludzie robią upgrade z jednej wersji do drugiej, powinno być bezboleśnie
    jasne kiedy coś się może zepsuć. Powinno być możliwe upgrade'owanie się do
    wersji, która wypisuje zdezaprobowania, usunięcie tego, co jest
    zdezaprobowane, następnie upgrade'owanie się do wersji, w której
    dezaprobowane funkcjonalności są usuwane.

  %p
    Jeśli nie robisz nic innego, wypisz w swoim changelogu zdezaprobowania,
    usunięcia i&nbsp;jakiekolwiek zmiany łamiące zgodność wstecz.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Mylące daty

  %p
    Regionalne formaty dat różnią się na świecie i często trudno jest znaleźć
    przyjazny człowiekowi format daty, który wydaje się intuicyjny wszystkim.
    Zaletą dat sformatowanych tak jak <code>2017-07-17</code> jest to, że są one
    uporządkowane od największych do najmniejszych jednostek: roku, miesiąca
    i dnia. Ten format również nie nachodzi w niejednoznaczny sposób na inne
    formaty dat, w przeciwieństwie do niektórych regionalnych formatów, które
    zamieniają miejsce numerów miesiąca i dnia. Z tych powodów i faktu, że
    ten format daty jest #{link_to "standardem ISO", data.links.isodate}, wynika rekomendacja
    tego formatu daty do wpisów w&nbsp;changelogu.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" } 
    Niespójne zmiany

  %p
    Changelog, który wspomina tylko o niektórych zmianach, może być równie 
    niebezpieczny jak brak changeloga. Chociaż wiele zmian może nie być istotnych 
    &mdash; na przykład usunięcie pojedynczej białej spacji może nie wymagać odnotowania 
    &mdash; to wszelkie ważne zmiany powinny być wymienione w changelogu.
    Poprzez niekonsekwentne stosowanie zmian, Twoi użytkownicy mogą błędnie myśleć, 
    że changelog jest jedynym źródłem prawdy. I tak być powinno. Siła wynika tuaj  
    z odpowiedzialności &mdash; posiadanie dobrego changeloga oznacza posiadanie konsekwentnie
    aktualizowanego changeloga.

  %aside
    Jest tego więcej. Pomóż mi zebrać te antywzorce
    = link_to "otwierając zgłoszenie", data.links.issue
    lub pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Często zadawane pytania

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Czy istnieje standardowy format changelogu?

  %p
    Niezupełnie. Jest przewodnik stylu changelogu GNU, czy dwuparagrafowe
    „wytyczne” GNU NEWS. Oba dokumenty są nieadekwatne lub niewystarczające.

  %p
    Ten projekt pretenduje do bycia
    = link_to "lepszą konwencją changelogu.", data.links.changelog
    Pochodzi z obserwowania i zebrania dobrych praktyk w społeczności open
    source.

  %p
    Zdrowa krytyka, dyskusja i sugestie poprawek
    = link_to "są mile widziane.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Jak powinien się nazywać plik z changelogiem?

  %p
    Nazwij go <code>CHANGELOG.md</code>. Niektóre projekty używają
    <code>HISTORY</code>, <code>NEWS</code> lub <code>RELEASES</code>.

  %p
    Łatwo jest uznać, że nazwa pliku z changelogiem nie ma większego znaczenia,
    lecz po co utrudniać swoim użytkownikom końcowym odnajdowanie w sposób
    konsekwentny istotnych zmian?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Co z GitHub Releases?

  %p
    To wspaniała inicjatywa. #{link_to "Releases", data.links.github_releases} mogą być używane do
    zmiany prostych tagów Gita (na przykład taga nazwanego <code>v1.0.0</code>)
    w bogate informacje o wydaniach przez ręczne dodanie informacji lub mogą
    wyciągać oznaczone message tagów i przekształcać je w informacje.

  %p
    GitHub Releases tworzą nieprzenośny changelog, który może być prezentowany
    użytkownikom tylko w kontekście GitHuba. Można go bardzo upodobnić do
    formatu Prowadź changelog, ale będzie to dość skomplikowane.

  %p
    Bieżąca wersja wydań GitHub jest też prawdopodobnie nienajłatwiejsza do
    odnalezienia dla użytkowników końcowych, w przeciwieństwie do plików
    o nazwach z wielkimi literami (<code>README</code>, <code>CONTRIBUTING</code>
    itp.). Innym mniejszym brakiem jest to, że interfejs obecnie nie posiada
    linków do logów commitów pomiędzy każdymi wydaniami.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Czy changelogi mogą być parsowane automatycznie?

  %p
    To trudne, ponieważ ludzie stosują bardzo różne formaty i nazwy plików.

  %p
    #{link_to "Vandamme", data.links.vandamme} jest gemem Ruby stworzonym przez zespół
    Gemnasium i który parsuje wiele (ale nie wszystkie)
    changelogów projektów open source.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Co z wycofanymi wydaniami?

  %p
    Wydania typu yanked to wersje, które musiały zostać usunięte z powodu
    poważnego błędu lub problemów z bezpieczeństwem. Często te wersje nie
    pojawiają się w dziennikach zmian. Powinny. Tak powinieneś je wyświetlać:

  %p <code>## 0.0.5 - 2014-12-13 [WYCOFANA]</code>

  %p
    Etykieta <code>[WYCOFANA]</code> jest celowo zapisana wielkimi literami.
    Ważne jest, by zwracano na nią uwagę. Jest otoczona nawiasami, więc jest
    również prostsza do sparsowania przez skrypt.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Czy powinno się kiedykolwiek przerabiać changelog?

  %p
    Pewnie. Zawsze istnieją dobre powody, by ulepszyć changelog. Regularnie
    otwieram pull requesty dodające brakujące wydania do open-source'owych
    projektów z nieutrzymywanymi changelogami.

  %p
    Może się również zdarzyć, że odkryjesz, iż zapomniałeś udokumentować zmianę
    zrywającą zgodność wsteczną w notatkach dla wersji. Oczywiście ważne jest,
    abyś zaktualizował swój changelog w tym przypadku.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Jak mogę wnieść swój wkład?

  %p
    Ten dokument nie jest obiektywną <strong>prawdą</strong>; jest moją
    starannie przemyślaną opinią, z informacjami i przykładami, które zebrałem.

  %p
    To dlatego, że chcę, aby nasza społeczność osiągnęła konsensus. Wierzę, że
    dyskusja jest tak samo istotna jak efekt końcowy.

  %p
    Więc proszę, <strong>#{link_to "wtrąć się", data.links.repo}</strong>.

.press
  %h3 Rozmowy
  %p
    Byłem na #{link_to "podcaście Changelog", data.links.thechangelog}, aby porozmawiać
    dlaczego opiekunowie i kontrybutorzy powinni dbać o changelogi, a także
    o motywacjach stojących za tym projektem.
```

## File: source/pt-BR/0.3.0/index.html.haml
```haml
---
title: Mantenha um CHANGELOG
description: Não deixe seus amigos despejarem logs de commits em CHANGELOGs
language: pt-BR
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### O que é um *changelog*?

  Um *changelog* é um arquivo que contém uma lista selecionada, ordenada
  cronologicamente, de mudanças significativas para cada versão de um projeto
  *open source*.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Para que serve um *changelog*?

  Para facilitar que usuários e contribuidores vejam precisamente quais
  mudanças significativas foram realizadas entre cada versão publicada.

  ### Por que eu deveria me importar?

  Porque softwares são feitos para pessoas. Se você não liga, por que está
  contribuindo com projetos *open source*? Certamente deve haver um fundo de
  carinho em algum lugar desse seu amável cerebrinho.

  Eu [conversei com Adam Stacoviak e Jerod Santo no podcast do The
  Changelog][thechangelog] (faz sentido, hein?) sobre por que mantenedores e
  contribuidores *open source* devem se importar e as motivações por trás desse
  projeto. Se você tem tempo (1:06), é um bom programa.

  ### O que faz um *changelog* ser bom?

  Que bom que perguntou.

  Um bom *changelog* se atém a esses princípios:

  - É feito para seres humanos, não máquinas, então legibilidade é crucial.
  - É fácil de referenciar (*linkar*) qualquer seção (por isso Markdown ao
    invés de puro texto).
  - Uma subseção por versão.
  - Lista as versões publicadas em ordem cronológica decrescente (mais novo em
    cima).
  - Usa todas as datas no formato `AAAA-MM-DD`. (Exemplo: `2012-06-02` para
    `2 de Junho de 2012`.) É internacional, [sensato](https://xkcd.com/1179/), e
    independente de língua.
  - Menciona explicitamente se o projeto segue o [Versionamento Semântico][semver].
  - Cada versão deve:
    - Listar sua data de publicação no formato acima.
    - Agrupar mudanças descrevendo seus impactos no projeto, como segue:
      - `Added`/`Adicionado` para novas funcionalidades.
      - `Changed`/`Modificado` para mudanças em funcionalidades existentes.
      - `Deprecated`/`Obsoleto` para funcionalidades estáveis que foram removidas das
        próximas versões.
      - `Removed`/`Removido` para funcionalidades removidas desta versão.
      - `Fixed`/`Consertado` para qualquer correção de bug.
      - `Security`/`Segurança` para incentivar usuários a atualizarem em caso de
        vulnerabilidades.

  ### Como eu posso minimizar o esforço exigido?

  Mantenha sempre uma seção `"Unreleased"`\`"A publicar"` no topo para manter o controle de
  quaisquer mudanças.

  Isso serve a dois propósitos:

  - As pessoas podem ver quais mudanças elas podem esperar em publicações
    futuras.
  - No momento da publicação, você apenas tem de mudar o `"Unreleased"`\`"A publicar"` para o
    número de versão e adicionar um novo cabeçalho `"Unreleased"`\`"A publicar"` no topo.

  ### O que faz os unicórnios chorarem?

  Tudo bem... vamos lá.

  - **Despejar logs de commits.** Simplesmente não faça isso, você não está
    ajudando ninguém.
  - **Não dar ênfase nas obsolências.** Quando as pessoas atualizam de uma versão
    para outra, deve ser incrivelmente claro quando algo irá parar de funcionar.
  - **Datas em formatos específicos de uma região.** Nos Estados Unidos, as pessoas
    colocam o mês primeiro ("06-02-2012" para 2 de Junho de 2012, o que *não*
    faz sentido), enquanto muitos no resto do mundo escrevem em aspecto
    robótico "2 Junho 2012", e mesmo assim leem de forma diferente.
    "2012-06-02" funciona de maneira lógica do maior para o menor, não
    se confunde de maneira ambígua com outros formatos, e é um [padrão
    ISO](http://www.iso.org/iso/home/standards/iso8601.htm).  Portanto, é o
    formato recomendado para *changelogs*.

  Tem mais. Ajude-me a coletar essas lágrimas de unicórnio [abrindo uma
  issue][issues] ou um *pull request*.

  ### Existe um padrão de formato de *changelog*?

  Infelizmente, não. Calma. Eu sei que você está buscando furiosamente
  aquele link do guia GNU de estilo de *changelog*, ou o arquivo "guideline"
  de dois paragráfos do GNU NEWS. O estilo GNU é um bom começo mas, infelizmente, é ingênuo.
  Não há nada de errado em ser ingênuo mas, quando as pessoas precisam de orientação,
  raramente ajuda. Especialmente quando existem muitas situações excepcionais
  para lidar.

  Este projeto [contém o que eu espero se tornar um melhor padrão de arquivo
  CHANGELOG][CHANGELOG] para todos os projetos *open source*. Eu não acredito que o status quo 
  seja bom o suficiente, e acho que, como uma comunidade, nós podemos encontrar novas convenções
  se tentarmos extrair boas práticas de projetos de software reais. Por favor, dê uma olhada e 
  lembre-se de que [discussões e sugestões de melhorias são bem-vindas][issues]!

  ### Qual deve ser o nome do arquivo *changelog*?

  Bom, se você não notou no exemplo acima, `CHANGELOG.md` é a melhor convenção até agora.

  Alguns outros projetos também utilizam `HISTORY.txt`, `HISTORY.md`,
  `History.md`, `NEWS.txt`, `NEWS.md`, `News.txt`, `RELEASES.txt`,
  `RELEASE.md`, `releases.md`, etc.

  É uma bagunça. Todos esses nomes só dificultam encontrar o *changelog*.

  ### Por que as pessoas não podem simplesmente utilizar `git log`?

  Porque os logs do Git são cheios de ruído — por natureza. Eles não serviriam como um 
  changelog adequado mesmo em um projeto hipotético executado por humanos perfeitos, que
  nunca erram uma letra, nunca esquecem de *commitar* arquivos, nunca cometem deslizes
  em uma refatoração. O propósito de um *commit* é documentar um passo atômico no 
  processo pelo qual o código evolui de um estado a outro. O propósito de um *changelog*
  é documentar as diferenças relevantes entre esses estados.
  
  A mesma diferença que existe entre bons comentários e o código em si existe entre o 
  *changelog* e o *commit log*: um descreve o *porquê*, o outro descreve o *como*.

  ### Podem os *changelogs* ser automaticamente interpretados?

  É difícil, porque as pessoas seguem formatos e nomes de arquivos
  radicalmente diferentes.

  [Vandamme][vandamme] é uma gem criada pelo time [Gemnasium][gemnasium] que
  interpreta muitos (mas não todos) *changelogs* de projetos *open source*.

  ### Por que você alterna entre as grafias "CHANGELOG" e "changelog"?

  "CHANGELOG" é o nome do arquivo em si. É um pouco chamativo mas é uma
  convenção histórica seguida por muitos projetos *open source*. Outros exemplos
  similares de arquivo incluem [`README`][README], [`LICENSE`][LICENSE], e
  [`CONTRIBUTING`][CONTRIBUTING].

  O nome em letras maiúsculas (o que em sistemas operacionais antigos faziam
  estes arquivos ficarem no topo da lista) é utilizado para chamar atenção.
  Por conterem importantes metadados do projeto, *changelogs* são úteis a
  qualquer um que pretenda utilizar ou contribuir com o projeto, da maneira
  equivalente às [badges de projetos *open source*][shields].

  Quando eu me refiro a "*changelog*", eu estou falando sobre a função desse
  arquivo: listar mudanças.

  ### E sobre as publicações removidas?

  Publicações removidas são versões que tiveram que ser removidas devido a um
  sério bug ou problema de segurança. Com frequência essas versões sequer
  aparecem em *changelogs*. Deveriam. É assim que você deve exibi-las:

  `## [0.0.5] - 2014-12-13 [YANKED]`
  
  A tag `[YANKED]`/`[REMOVIDA]` é chamativa por uma razão. É importante que
  as pessoas a notem. Além disso, já que é cercada por colchetes, é mais
  fácil detectá-la programaticamente.

  ### Devemos alterar o *changelog* em alguma circunstância?
 
  Claro. Sempre existem bons motivos para melhorar um *changelog*. Eu regularmente
  abro *pull requests* para adicionar versões faltantes em projetos *open source*
  com *changelogs* abandonados.

  Também é possível que você perceba que se esqueceu de registrar mudanças
  importantes nas notas de uma versão. É obviamente importante que você atualize
  seu *changelog* nesse caso.

  ### Como eu posso contribuir?

  Este documento não é a verdade; é minha cuidadosa opinião, junto com as
  informações e exemplos que reuni. Embora eu tenha providenciado um
  [CHANGELOG][] no [repositório no GitHub][gh], eu não criei de propósito uma
  lista clara de regras a serem seguidas (como o [SemVer.org][semver] faz, por
  exemplo).

  Fiz isso porque eu gostaria que nossa comunidade chegasse a um consenso. Eu
  acredito que a discussão é tão importante quanto o resultado final.

  Então, por favor, [entre em campo][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/pt-BR/1.0.0/index.html.haml
```haml
---
title: Mantenha um Changelog
description: Não deixe seus amigos despejarem logs de commits no Changelog
language: pt-BR
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    O que é um changelog?

  %p
    Um changelog é um arquivo que contém uma lista selecionada, ordenada
    cronologicamente, de mudanças significativas para cada versão de um projeto.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Por que manter um changelog?

  %p
    Para facilitar que usuários e contribuidores vejam precisamente quais
    mudanças significativas foram realizadas entre cada versão publicada de
    um projeto.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Quem precisa de um changelog?

  %p
    Pessoas precisam. Seja consumidores ou desenvolvedores,
    os usuários finais de softwares são seres humanos
    que se preocupam com o que está no software. Quando
    o software muda, as pessoas querem saber por que e como.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Como fazer um bom changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Princípios fundamentais

  %ul
    %li
      Changelogs são <em>para humanos</em>, não máquinas.
    %li
      Deve haver uma entrada para cada versão.
    %li
      Alterações do mesmo tipo devem ser agrupadas.
    %li
      Versões e seções devem ser vinculáveis (com links).
    %li
      A versão mais recente vem em primeiro lugar.
    %li
      A data de lançamento de cada versão é exibida.
    %li
      Mencione se você segue o #{link_to "versionamento semântico", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipos de mudanças

  %ul
    %li
      %code Added/Adicionado
      para novos recursos.
    %li
      %code Changed/Modificado
      para alterações em recursos existentes.
    %li
      %code Deprecated/Obsoleto
      para recursos que serão removidos nas próximas versões.
    %li
      %code Removed/Removido
      para recursos removidos nesta versão.
    %li
      %code Fixed/Corrigido
      para qualquer correção de bug.
    %li
      %code Security/Segurança
      em caso de vulnerabilidades.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Como eu posso minimizar o esforço exigido para manter um changelog?

  %p
    Mantenha sempre uma seção <code>Unreleased/Não publicado</code> no topo para manter o controle das novas mudanças.

  %p Isso serve a dois propósitos:

  %ul
    %li
      As pessoas podem ver quais mudanças elas podem esperar em publicações futuras.
    %li
      No momento da publicação, você apenas tem de mudar a seção
      <code>Unreleased/Não publicado</code> para o número de versão e adicionar uma
      nova seção <code>Unreleased/Não publicado</code> no topo.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Os changelogs podem ser ruins?

  %p Sim. Aqui estão algumas maneiras pelas quais eles podem ser inúteis.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Usar um registro de alterações automático

  %p
    Usar um registro de alterações automático é uma má ideia: eles estão
    cheios de bagunça. Coisas como solicitação de mesclagem, envio com títulos
    estranhos, alterações de documentação, etc.

  %p
    O propósito de um commit é documentar a etapa na evolução do código
    fonte. Alguns projetos limpam os commits, outros não.

  %p
    O propósito de uma entrada de changelog é documentar as diferenças
    notáveis, muitas vezes de múltiplos commits, para comunicar de forma
    clara os usuários.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorando depreciações

  %p
    Quando pessoas atualizam de uma versão para outra, deve ficar muitíssimo claro
    quando algo vai quebrar. Deve ser possível atualizar para uma versão
    com depreciações listadas, remova o que é obsoleto, depois atualize
    para a versão onde as depreciações se tornam remoções.

  %p
    Se você não fizer mais nada, liste no seu changelog as depreciações,
    remoções e quaisquer mudanças que gerem falhas.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Datas confusas

  %p
    Os formatos regionais de data variam em todo o mundo e muitas vezes
    é difícil encontrar um formato de data amigável que seja intuitivo para todos.
    A vantagem das datas formatadas como <code>2017-07-17</code> é que elas seguem
    a ordem da maior para a menor unidade de tempo: ano, mês e dia. Este formato
    também não se confunde de maneira ambígua com outros formatos de data, ao
    contrário de alguns formatos regionais que alteram a posição dos números do mês
    e dia. Esses motivos, e o fato de ser um formato de data suportado pela
    #{link_to "norma ISO", data.links.isodate} são as razões para ele ser o formato de data
    recomendado para as entradas do changelog.

  %aside
    Tem mais. Me ajude a colecionar essas más práticas
    = link_to "enviando uma dúvida", data.links.issue
    ou pedindo mudanças.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Perguntas frequentes

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Existe um padrão para o formato do changelog?

  %p
    Na verdade não. Existe o guia de estilo de changelog do GNU
    ou o "guia" de dois parágrafos do GNU NEWS. Ambos são inadequados ou
    insuficientes.

  %p
    Este projeto pretende ser #{link_to "uma convenção de changelog melhor.", data.links.changelog}
    Ele vem de observar e coletar as boas práticas em código aberto da
    comunidade.

  %p
    Críticas saudáveis, discussões e sugestões de melhorias
    = link_to "são bem-vindas.", data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Qual nome o arquivo changelog deve ter?

  %p
    Chame-o <code>CHANGELOG.md</code>. Alguns projetos usam
    <code>HISTORY</code>, <code>NEWS</code> ou <code>RELEASES</code>.

  %p
    Embora seja fácil pensar que o nome do seu arquivo changelog
    não importa muito, por que tornar mais difícil para seus usuários
    finais encontrarem consistentemente mudanças notáveis?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    E sobre o GitHub Releases?

  %p
    É uma grande iniciativa. #{link_to "Lançamentos", data.links.github_releases} podem ser usados
    para converter simples marcadores do git (por exemplo, um
    marcador chamado <code>v1.0.0</code>) em notas de versão ricas,
    adicionando manualmente notas de versão ou pode puxar as mensagens
    anotadas no marcador do git e transformá-las em notas.

  %p
    GitHub Releases cria um changelog não portátil
    que só pode ser exibido para usuários no contexto do GitHub.
    É possível fazê-los parecer muito como o formato
    Keep a Changelog, mas tende a ser um pouco mais complicado.

  %p
    A versão atual do GitHub Releases não é facilmente descoberta
    por usuários finais, ao contrário dos arquivos maiúsculos típicos
    (<code>README</code>, <code>CONTRIBUTING</code>, etc.). Outro
    problema de menor magnitude é que a interface atualmente não oferece
    links para confirmar alterações entre cada lançamento.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Os changelogs podem ser criados automaticamente?

  %p
    É difícil, porque as pessoas seguem formatos e nomes de arquivos
    totalmente diferentes.

  %p
    #{link_to "Vandamme", data.links.vandamme} é um gem Ruby criado pelo
    time Gemnasium e que analisa muitas
    (mas não todas) alterações de projetos de código aberto.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    E o lançamentos removidos?

  %p
    Lançamentos removidos são versões que foram retiradas por causa de
    problemas sérios ou falhas de segurança. Muitas vezes essas versões
    nem aparecem no histórico de alterações. Eles deviam. É assim que
    você deve exibi-los:

  %p <code>## 0.0.5 - 2014-12-13 [REMOVIDO]</code>

  %p
    O marcador <code>[REMOVIDO]</code> está em caixa alta por uma razão.
    É importante que as pessoas o percebam. Já que está entre
    colchetes, também fica mais fácil de ser analisado programaticamente.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Você deve reescrever um changelog?

  %p
    Claro. Sempre existe razão para melhorar um changelog.
    Eu regularmente solicito alterações em projetos de código livre que
    possuem changelogs não mantidos para adicionar lançamentos
    que não estão presentes nestes.

  %p
    Também é possível que você descubra que você esqueceu de abordar
    uma mudança abrupta nas notas para uma versão.
    Obviamente é importante que você atualize seu changelog neste caso.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Como eu posso ajudar?

  %p
    Esse documento não é uma <strong>verdade absoluta</strong>; É minha
    opinião cuidadosamente considerada, juntamente com informações e
    exemplos que eu reuni.

  %p
    Isso porque o que eu quero é que nossa comunidade chegue a um consenso.
    Eu acredito que a discussão é tão importante quanto o resultado final.

  %p
    Então, por favor <strong>#{link_to "contribua", data.links.repo}</strong>.

.press
  %h3 Discussões
  %p
    Eu fui no #{link_to "The Changelog podcast", data.links.thechangelog}
    para falar sobre por que os mantenedores e contribuidores devem se
    preocupar com os changelogs, e também sobre as motivações
    por trás desse projeto.
```

## File: source/pt-BR/1.1.0/index.html.haml
```haml
---
title: Mantenha um Changelog
description: Não deixe seus amigos despejarem logs de commits no Changelog
language: pt-BR
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    O que é um changelog?

  %p
    Um changelog é um arquivo que contém uma lista selecionada, ordenada
    cronologicamente, de mudanças significativas para cada versão de um projeto.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Por que manter um changelog?

  %p
    Para facilitar que usuários e contribuidores vejam precisamente quais
    mudanças significativas foram realizadas entre cada versão publicada de
    um projeto.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Quem precisa de um changelog?

  %p
    Pessoas precisam. Seja consumidores ou desenvolvedores,
    os usuários finais de softwares são seres humanos
    que se preocupam com o que está no software. Quando
    o software muda, as pessoas querem saber por que e como.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Como fazer um bom changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Princípios fundamentais

  %ul
    %li
      Changelogs são <em>para humanos</em>, não máquinas.
    %li
      Deve haver uma entrada para cada versão.
    %li
      Alterações do mesmo tipo devem ser agrupadas.
    %li
      Versões e seções devem ser vinculáveis (com links).
    %li
      A versão mais recente vem em primeiro lugar.
    %li
      A data de lançamento de cada versão é exibida.
    %li
      Mencione se você segue o #{link_to "versionamento semântico", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipos de mudanças

  %ul
    %li
      %code Adicionado
      para novas funcionalidades.
    %li
      %code Modificado
      para alterações em funcionalidades existentes.
    %li
      %code Obsoleto
      para funcionalidades que estão para ser removidas.
    %li
      %code Removido
      para funcionalidades removidas nesta versão.
    %li
      %code Corrigido
      para qualquer correção de bug.
    %li
      %code Segurança
      em caso de vulnerabilidades.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Como eu posso minimizar o esforço exigido para manter um changelog?

  %p
    Mantenha sempre uma seção <code>Não publicado</code> no topo para rastrear alterações
    vindouras.

  %p Isso serve a dois propósitos:

  %ul
    %li
      As pessoas podem ver quais mudanças elas podem esperar em publicações futuras.
    %li
      No momento da publicação, você pode mover a seção de mudanças <code>Não publicado</code>
      como uma seção para uma nova publicação.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Os changelogs podem ser ruins?

  %p Sim. Aqui estão algumas maneiras pelas quais eles podem ser inúteis.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffs de registros de commits

  %p
    Utilizar diffs de registros de commits como changelogs é uma má ideia: eles estão cheios de
    bagunça. Coisas como commits de mesclagem, commits com títulos estranhos,
    alterações de documentação etc.

  %p
    O propósito de um commit é documentar a etapa na evolução do código
    fonte. Alguns projetos limpam os commits, outros não.

  %p
    O propósito de uma entrada de changelog é documentar as diferenças
    notáveis, muitas vezes de múltiplos commits, para comunicá-las de forma
    clara aos usuários.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorando descontinuações

  %p
    Quando pessoas atualizam de uma versão para outra, deve ficar
    muitíssimo claro quando algo vai quebrar. Deve ser possível atualizar
    para uma versão que liste descontinuações, remova o que foi descontinuado,
    e então atualize para a versão em que descontinuações foram removidas.

  %p
    Se você não fizer mais nada, liste no seu changelog as depreciações,
    remoções e quaisquer mudanças que gerem falhas.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Datas confusas

  %p
    Os formatos regionais de data variam em todo o mundo e muitas vezes
    é difícil encontrar um formato de data amigável que seja intuitivo para todos.
    A vantagem das datas formatadas como <code>2017-07-17</code> é que elas seguem
    a ordem da maior para a menor unidade de tempo: ano, mês e dia. Este formato
    também não se confunde de maneira ambígua com outros formatos de data, ao
    contrário de alguns formatos regionais que alteram a posição dos números do mês
    e dia. Esses motivos, e o fato de ser um formato de data suportado pela
    #{link_to "norma ISO", data.links.isodate} são as razões para ele ser o formato de data
    recomendado para as entradas do changelog.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Alterações Inconsistentes

  %p
    Um changelog que apenas menciona algumas das alterações pode ser tão perigoso
    quanto não ter um changelog. Enquanto muitas das alterações talvez não sejam
    relevantes - como por exemplo remover um único espaço em branco talvez não necessite
    ser registrado todas as vezes - qualquer alteração importante deve ser
    mencionada no changelog. Por registrar alterações de forma inconsistente,
    seus usuários podem pensar, incorretamente, que o changelog é a fonte única de verdade.
    Deveria ser. Com grandes poderes vem grandes responsabilidade -
    ter um bom changelog siginifica ter um changelog consistentemente atualizado.

  %aside
    Tem mais. Me ajude a colecionar essas más práticas
    = link_to "enviando uma dúvida", data.links.issue
    ou um pedindo mudanças.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Perguntas frequentes

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Existe um padrão para o formato do changelog?

  %p
    Na verdade não. Existe o #{link_to "guia de estilo de changelog do GNU", data.links.gnustyle},
    ou o "guia" #{link_to "de dois parágrafos do GNU NEWS file", data.links.gnunews}.
    Ambos são inadequados ou insuficientes.

  %p
    Este projeto pretende ser
    = link_to "uma convenção para um changelog melhor.", data.links.changelog
    Ele vem da observação de boas práticas na comunidade de código aberto
    e a reunião delas.

  %p
    Críticas saudáveis, discussões e sugestões de melhorias
    = link_to "são bem-vindas.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Qual nome o arquivo de changelog deve ter?

  %p
    Chame-o <code>CHANGELOG.md</code>. Alguns projetos usam
    <code>HISTORY</code>, <code>NEWS</code> ou <code>RELEASES</code>.

  %p
    Embora seja fácil pensar que o nome do seu arquivo de changelog
    não importa muito, por que tornar mais difícil para seus usuários
    finais encontrarem consistentemente mudanças notáveis?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    E sobre o GitHub Releases?

  %p
    É uma grande iniciativa. #{link_to "Lançamentos", data.links.github_releases} podem ser usados
    para converter simples marcadores do git (por exemplo um marcador chamada <code>v1.0.0</code>)
    em ricas anotações de lançamento, adicionando manualmente anotações de lançamento ou pode
    puxar as mensagens anotadas no marcador do git e transformá-las em notas.

  %p
    GitHub Releases cria um changelog não portátil
    que só pode ser exibido para usuários no contexto do GitHub.
    É possível fazê-los parecer muito como o formato
    Keep a Changelog, mas tende a ser um pouco mais complicado.

  %p
    É possível argumentar também que a versão atual do GitHub releases não é de fácil
    descoberta por usuários finais, ao contrário dos típicos arquivos em maiúsculo
    (<code>README</code>, <code>CONTRIBUTING</code> etc.). Outra questão é que a
    interface atualmente não oferece ligação (com links) para registros de commits
    entre cada lançamento.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Os changelogs podem ser criados automaticamente?

  %p
    É difícil, porque as pessoas seguem formatos e nomes de arquivos
    totalmente diferentes.

  %p
    #{link_to "Vandamme", data.links.vandamme} é um gem Ruby criado pelo
    time Gemnasium e que analisa muitos (mas
    não todos) changelogs de projetos de código aberto.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    E sobre lançamentos removidos?

  %p
    Lançamentos removidos são versões que foram retiradas por causa de
    problemas sérios ou falhas de segurança. Muitas vezes essas versões
    nem aparecem no histórico de alterações. Eles deviam. É assim que
    você deve exibi-los:

  %p <code>## [0.0.5] - 2014-12-13 [REMOVIDO]</code>

  %p
    O marcador <code>[REMOVIDO]</code> está em caixa alta por uma razão.
    É importante que as pessoas o percebam. Já que está entre colchetes
    fica mais fácil ser analisado programaticamente também.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Você deve reescrever um changelog?

  %p
    Claro. Sempre existe uma boa razão para melhorar um changelog.
    Eu regularmente solicito alterações para adicionar lançamentos faltantes
    em projetos de código aberto que não mantém um changelog.

  %p
    Também é possível que você descubra que você esqueceu de abordar
    uma mudança abrupta nas anotações para uma versão.
    Obviamente é importante que você atualize seu changelog neste caso.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Como eu posso ajudar?

  %p
    Esse documento não é uma <strong>verdade absoluta</strong>; É minha
    opinião cuidadosamente considerada, juntamente com informações e exemplos
    que eu reuni.

  %p
    Isso porque o que eu quero é que nossa comunidade chegue a um consenso.
    Eu acredito que a discussão é tão importante quanto o resultado final.

  %p
    Então, por favor <strong>#{link_to "contribua", data.links.repo}</strong>.

.press
  %h3 Discussões
  %p
    Eu fui no #{link_to "The Changelog podcast", data.links.thechangelog}
    para falar sobre por que os mantenedores e contribuidores deveriam se preocupar
    com os changelogs, e também sobre as motivações por trás desse projeto.
```

## File: source/ro/1.1.0/index.html.haml
```haml
---
title: Ține un jurnal de modificări
description: Nu-ți lăsa prietenii să arunce jurnalele git în jurnalul de modificări.
language: ro
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Versiune
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Ce este un jurnal de modificări?

  %p
    Un jurnal de modificări este un fișier care conține o listă
    organizată, ordonată cronologic de modificări notabile pentru
    fiecare versiune a unui proiect.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    De ce să ții un jurnal de modificări?

  %p
    Pentru ca utilizatorii și colaboratorii să vadă exact ce
    modificări notabile au fost făcute între fiecare lansare (sau
    versiune) a proiectului.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Cine are nevoie de un jurnal de modificări?

  %p
    Oamenii. Indiferent dacă sunt consumatori sau dezvoltatori,
    utilizatorii finali ai software-ului sunt ființe umane cărora le pasă
    de ceea ce este în software. Când software-ul se schimbă, oamenii vor
    să știe de ce și cum.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Cum fac un jurnal de modificări bun?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Principii îndrumătoare

  %ul
    %li
      Jurnalele de modificări sunt <em>pentru oameni</em>, nu mașini.
    %li
      Ar trebui să existe o intrare pentru fiecare versiune.
    %li
      Aceleași tipuri de modificări ar trebui grupate.
    %li
      Versiunile și secțiunile ar trebui să poată fi legate.
    %li
      Cea mai recentă versiune este pe primul loc.
    %li
      Este afișată data de lansare a fiecărei versiuni.
    %li
      Menționează dacă urmărești #{link_to "Numerotarea Semantică", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Tipuri de modificări

  %ul
    %li
      %code Added
      pentru funcționalități noi.
    %li
      %code Changed
      pentru modificări în funcționalitatea existentă.
    %li
      %code Deprecated
      pentru funcționalități ce urmează să fie eliminate în curând.
    %li
      %code Removed
      pentru funcționalități eliminate acum.
    %li
      %code Fixed
      pentru orice erori remediate.
    %li
      %code Security
      în caz de vulnerabilități.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Cum pot reduce efortul necesar pentru a menține un jurnal de modificări?

  %p
    Ține o secțiune <code>Unreleased</code> în partea de sus pentru a
    urmări modificările viitoare.

  %p Aceasta servește la două scopuri:

  %ul
    %li
      Oamenii pot vedea la ce schimbări s-ar putea aștepta în lansările viitoare
    %li
      În momentul lansării, poți muta modificările din secțiunea
      <code>Unreleased</code> într-o nouă secțiune a versiunii de
      lansare.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Jurnalele de modificări pot fi precare?

  %p Da. Iată câteva moduri în care pot fi mai puțin decât utile.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Comitere jurnal de diferențe

  %p
    Utilizarea jurnalului de diferențe din comiteri ca jurnal de
    modificări este o idee proastă: sunt pline de zgomot. Lucruri
    precum comiterile de îmbinare, comiterile cu titluri obscure,
    modificările documentației, etc.

  %p
    Scopul unei comiteri este de a documenta un pas în evoluția
    codului sursă. Unele proiecte curăță comiterile, altele nu.

  %p
    Scopul unei intrări din jurnalul de modificări este de a
    documenta diferența demnă de remarcat, adesea din mai multe
    comiteri, pentru a le comunica în mod clar utilizatorilor
    finali.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorarea deprecierilor

  %p
    Când oamenii fac actualizare de la o versiune la alta, ar
    trebui să fie foarte clar când ceva nu va funcționa. Ar trebui să
    fie posibil să actualizezi la o versiune care listează
    deprecierile, să elimini ceea ce este depreciat, apoi să
    actualizezi la versiunea în care deprecierile devin eliminări.

  %p
    Dacă nu faci nimic altceva, enumeră deprecierile, eliminările și
    orice modificări nerespective în jurnalul tău de modificări.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Date confuze

  %p
    Formatele regionale de date variază în întreaga lume și este adesea dificil să
    găsești un format de dată prietenos pentru oameni, care să pară intuitiv pentru
    toată lumea. Avantajul datelor formatate ca <code>2017-07-17</code> este că
    urmează ordinea de la unitățile mai mari la cele mai mici: an, lună și zi. De
    asemenea, acest format nu se suprapune în moduri ambigue cu alte formate de dată,
    spre deosebire de unele formate regionale care schimbă poziția numerelor lunii și
    zilei.
    Aceste motive și faptul că acest format de dată este un
    #{link_to "standard ISO", data.links.isodate}, este motivul pentru care este
    formatul de dată recomandat pentru intrările din jurnalul de modificări.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Modificări inconsecvente

  %p
    Un jurnal de modificări care menționează doar unele dintre modificări poate
    fi la fel de periculos ca și lipsa unui jurnal de modificări. Deși multe
    dintre modificări pot să nu fie relevante - de exemplu, eliminarea unui
    singur spațiu alb poate să nu fie nevoie să fie înregistrată în toate
    cazurile - orice modificări importante ar trebui menționate în jurnalul
    de modificări. Prin aplicarea inconsecventă a modificărilor, utilizatorii
    tăi pot crede în mod eronat că jurnalul de modificări este singura sursă
    de adevăr. Așa ar trebui să fie. Cu o mare putere vine o mare responsabilitate -
    a avea un jurnal de modificări bun înseamnă a avea un jurnal de modificări
    actualizat constant.

  %aside
    Mai sunt. Ajută-mă să colectez aceste antimodele
    = link_to "deschizând o problemă", data.links.issue
    sau o cerere de extragere.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Întrebări frecvente

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Există un format standard de jurnal de modificări?

  %p
    Nu chiar. Există #{link_to "Ghidul de stil pentru jurnalul de modificări GNU", data.links.gnustyle},
    sau „ghidul” #{link_to "GNU NEWS de două paragrafe", data.links.gnunews}
    Ambele sunt inadecvate sau insuficiente.

  %p
    Acest proiect își propune să fie
    = link_to "o convenție mai bună pentru jurnalul de modificări.", data.links.changelog
    El vine din observarea bunelor practici în comunitatea open source și adunarea lor.

  %p
    = link_to "Sunt binevenite", data.links.issue
    criticile sănătoase, discuțiile și sugestiile de îmbunătățire.


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Cum ar trebui să fie numit fișierul jurnal de modificări?

  %p
    Numește-l <code>CHANGELOG.md</code>. Unele proiecte folosesc
    <code>HISTORY</code>, <code>NEWS</code> sau <code>RELEASES</code>.

  %p
    Deși este ușor de crezut că numele fișierului jurnal de modificări
    nu contează atât de mult, de ce este mai greu pentru utilizatorii
    finali să găsească în mod constant modificări notabile?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Ce zici despre Lansările GitHub?

  %p
    Este o inițiativă grozavă. #{link_to "Lansări", data.links.github_releases} poate fi
    folosit pentru a transforma etichetele git simple (de exemplu, o etichetă numită
    <code>v1.0.0</code>) în note de lansare bine detaliate, adăugând manual note de
    lansare sau putând trage mesaje adnotate din etichetele git și transofmându-le în note.

  %p
    Lansările GitHub creează un jurnal de modificări non-portabil care
    poate fi afișat numai utilizatorilor în contextul GitHub. Este
    posibil să le faci să semene foarte mult cu formatul Ține un
    jurnal de modificări, dar tinde să fie puțin mai implicat.

  %p
    De asemenea, versiunea actuală a Lansărilor GitHub este probabil
    puțin descoperită de utilizatorii finali, spre deosebire de
    fișierele tipice cu majuscule (<code>README</code>,
    <code>CONTRIBUTING</code>, etc.). O altă problemă minoră este că
    interfața nu oferă momentan legături pentru a comite jurnale între
    fiecare lansare.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Jurnalele de modificări pot fi interpretate automat?

  %p
    Este dificil, pentru că oamenii urmează formate și nume de fișiere
    extrem de diferite.

  %p
    #{link_to "Vandamme", data.links.vandamme} este un gem Ruby creat
    de echipa Gemnasium și care interpretează multe (dar nu toate)
    jurnalele de modificări ale proiectelor open source.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Ce zici despre lansările smulse („yanked”)?

  %p
    Lansările smulse (de tip „yanked”) sunt versiuni care au trebuit să
    fie eliminate din cauza unei erori grave sau a unei probleme de
    securitate. Adesea, aceste versiuni nici măcar nu apar în jurnalele
    de modificări. Ar trebui. Iată cum ar trebui să le afișezi:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Eticheta <code>[YANKED]</code> este puternică dintr-un motiv. Este
    important pentru oameni să o observe. Deoarece este înconjurată de
    paranteze, este, de asemenea, mai ușor de interpretat programatic.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Ar trebui să rescrii vreodată un jurnal de modificări?

  %p
    Sigur. Există întotdeauna motive bune pentru a îmbunătăți un jurnal
    de modificări. Eu deschid în mod regulat solicitări de extragere
    pentru a adăuga versiunile lipsă proiectelor open source cu jurnalele
    de modificări neîntreținute.

  %p
    De asemenea, este posibil să descoperi că ai uitat să abordezi o
    modificare care afectează funcționalitatea în notele pentru o
    versiune. În mod evident, este important să actualizezi jurnalul de
    modificări în acest caz.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Cum pot contribui?

  %p
    Acest document nu este <strong>adevărul</strong>; este opinia mea
    atent analizată, împreună cu informațiile și exemplele pe care
    le-am adunat.

  %p
    Asta pentru că vreau ca comunitatea noastră să ajungă la un consens.
    Cred că discuția este la fel de importantă ca și rezultatul final.

  %p
    Așa că te rog <strong>#{link_to "să te prezinți", data.links.repo}</strong>.

.press
  %h3 Conversații
  %p
    Am mers la #{link_to "podcastul The Changelog", data.links.thechangelog}
    pentru a vorbi despre motivul pentru care întreținătorilor și colaboratorilor ar
    trebui să le pese de jurnalele de modificări și, de asemenea, despre motivațiile
    din spatele acestui proiect.
```

## File: source/ru/0.3.0/index.html.haml
```haml
---
title: Ведите Changelog
description: Не позволяйте друзьям сливать логи гита в CHANGELOG
language: ru
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Что такое лог изменений?

  Лог изменений – это файл, который содержит поддерживаемый, упорядоченный в
  хронологическом порядке список значимых изменений для каждой версии проекта с
  открытым исходным кодом.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Для чего нужен лог изменений?

  Чтобы пользователи и контрибьюторы могли с меньшими усилиями точно
  определить, какие значимые изменения были сделаны в каждом релизе (или
  версии) проекта.

  ### Почему я вообще должен думать об этом?

  Потому, что инструменты программирования делаются для людей. Если вам пофигу,
  зачем вы вообще занимаетесь программным обеспечением с открытым исходным
  кодом? У вас обязательно в голове должен быть центр «не пофигу» :)

  Я [беседовал с Адамом Стаковиаком и с Джеродом Санто в подкасте The
  Changelog][thechangelog] (в тему название, правда?) о том почему авторам
  программного обеспечения с открытым исходным кодом и их коллегам должно быть
  не пофигу, и о моих мотивах в создании этого проекта.  Послушайте, если у вас
  есть время (1:06).

  ### Что должно быть в хорошем логе изменений?

  Я рад, что вы спросили.

  Хороший лог изменений придерживается этих приниципов:

  - Он сделан для людей, а не машин, так что понятность имеет решающее
    значение.
  - Легко сослаться на любой раздел (поэтому Markdown лучше обычного текста).
  - Один подраздел на каждую версию.
  - Релизы перечислены в обратном хронологическом порядке (самые новые –
    сверху).
  - Пишите все даты в формате `YYYY-MM-DD`. (Например: `2012-06-02` для даты `2
    июня 2012`.) Он международный, [рациональный](https://xkcd.com/1179/), и
    независим от языка.
  - Ясно указывает, использует ли проект [Семантическое
    Версионирование][semver].
  - Каждая версия должна:
    - Показывать дату релиза в формате, упомянутом выше.
    - Группировать изменения согласно их влиянию на проект следующим образом:
      - `Added` для новых функций.
      - `Changed` для изменений в существующей функциональности.
      - `Deprecated` для функциональности, которая будет удалена в следующих
        версиях.
      - `Removed` для функциональности, которая удалена в этой версии.
      - `Fixed` для любых исправлений.
      - `Security` для обновлений безопасности.

  ### Как минимизировать время, необходимое для ведения лога изменений?

  Всегда ведите секцию `Unreleased` в начале файла, чтобы держать в ней не
  выпущенные изменения.

  Это нужно для двух вещей:

  - Люди смогут видеть, какие изменения ожидаются в ближайших релизах
  - В момент релиза вам нужно всего лишь заменить секцию `Unreleased` на номер
    версии и добавить новую секцию `Unreleased` в начале файла.

  ### Что заставляет плакать единорогов?

  Хорошо… давайте разберёмся.

  - **Выгрузка изменений лога коммитов.** Просто не делайте так, это никому не
    принесёт пользы.
  - **Отсутствие отметок планируемой к удалению функциональности.** Когда люди
    обновляются от одной версии к другой, им должно быть очевидно до боли, что
    сломается.
  - **Даты в местном формате.** В Соединённых Штатах, люди сначала пишут месяц
    («06-02-2012» для даты 2 июня 2012, что не имеет *никакого* смысла), хотя
    много людей в остальном мире пишет роботоподобное «2 июня 2012», всё ещё
    произнося это по-другому. «2012-06-02» логично работает от самого большого
    к самому маленькому, не пересекается с другими форматами дат, и является
    [стандартом ISO](http://www.iso.org/iso/home/standards/iso8601.htm). Таким
    образом, этот формат является рекомендуемым для логов изменений.

  Существуют и другие. Помогите мне собрать слёзы единорогов,
  [открыв тикет][issues] или пулл-реквест.

  ### Существует стандарт формата лога изменений?

  К сожалению, нет. Спокойней. Я знаю, что вы яростно бросились на поиски
  ссылки на GNU-руководства по стилю лога изменений, или на поиски файла
  «guideline» с парой параграфов в GNU NEWS. GNU-руководство по стилю неплохо
  для начала, но оно наивно.  В наивности нет ничего плохого, но когда людям
  нужно руководство, она редко бывает полезна.  Особенно, когда приходиться
  сталкиваться со множеством краевых случаев.

  Этот проект [содержит информацию, которая, я надеюсь, станет соглашением
  получше о ведении файлов CHANGELOG][CHANGELOG] для всех проектов с открытым
  исходным кодом. Может ли сообщество учиться на своих ошибках, а не просто
  действовать согласно десяти записям, которые были написаны много лет назал,
  и, якобы, без одной ошибки? Хорошо. Поэтому, пожалуйста, посмотрите вокруг, и
  помните, что [обсуждения и предложения по улучшению приветствуются][issues]!

  ### Как можно назвать файл лога изменений?

  Ну, если вы не не можете ответить на этот вопрос, глядя на пример выше,
  `CHANGELOG.md` пока наилучший вариант.

  Некоторые проекты используют `HISTORY.txt`, `HISTORY.md`, `History.md`,
  `NEWS.txt`, `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`,
  `releases.md`, и так далее.

  Это непорядок. Все эти имена только усложняют поиск нужного файла.

  ### Почему люди не могут использовать просто дифф команды `git log`?

  Потому, что диффы логов по своей природе содержат слишком много «шума». С их
  помощью невозможно сделать подходящий лог изменений даже в гипотетическом
  проекте, который делается идеальными программистами, которые никогда не
  делают опечаток, никогда не забывают коммитить новые файлы, никогда не
  пропускают ни одну часть рефакторинга.  Назначение коммита в том, чтобы
  задокументировать один атомарный шаг в процессе развития кода от одного
  состояния к другому. Назначение лога изменений – документировать значимые
  различия между этими состояниями.

  Как отличаются хорошие комментарии к коду и сам код, также отличаются друг от
  друга и лог изменений с логом коммитов: первые отвечают на вопрос
  *почему*, а вторые на вопрос как.

  ### Могут ли логи изменений быть автоматически распарсены?

  Это сложно из-за того, что люди следуют сильно отличающимся форматам и
  соглашениям о именах файлов.

  Гем для Руби [Vandamme][vandamme], который создали в команде
  [Gemnasium][gemnasium], парсит многие (но не все) логи изменений проектов с
  открытым исходным кодом.

  ### Почему вы пишите то «CHANGELOG», то «лог изменений»?

  «CHANGELOG» – это имя файла. Оно несколько крикливо, но это историческое
  соглашение, которому следуют многие проекты с открытым исходным кодом. В
  качестве примеров подобных файлов можно привести [`README`][README],
  [`LICENSE`][LICENSE], и [`CONTRIBUTING`][CONTRIBUTING].

  Верхний регистр в имени файла (который в старых операционных системах
  заставляет эти файлы находиться наверху списков) используется для привлечения
  внимания. Так как в этих файлах содержится важная метаинформация о проекте,
  они могут быть полезны любому использующему или дорабатывющему проект, также
  как [бейджи проектов с открытым исходным кодом][shields].

  Когда я упоминаю «лог изменений», я говорою о назначении этого файла: об
  учёте изменений.

  ### Как насчёт yanked-релизов?

  Yanked-релизы – это версии, которые изымаются из обращения из-за серьёзного
  бага или проблемы безопасности в них. Часто такие версии даже не упоминаются
  в логах изменений. А должны. И вот как вам следует их упоминать:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Тег `[YANKED]` такой «громкий» не просто так. Очень важно, чтобы люди его
  заметили. А из-за того, что он окружён скобками, его легче определить
  программно.

  ### Стоит ли переписывать лог изменений?

  Конечно. Всегда есть причины для улучшения лога изменений. Я регулярно
  открываю пулл-реквесты в проекты с открытым исходным кодом с
  неподдерживаемыми логами изменений для добавления недостающих релизов.

  Также вы можете обнаружить что вы забыли адресовать критичное изменение в
  описании версии. В этом случае очевидно, что нужно обновить лог
  изменений.

  ### Как я могу помочь?

  Этот документ **не истина в последней инстанции**; это моё тщательно
  сформулированное мнение вместе с информацией и примерами, которые я собрал.
  Хотя я предоставил настоящий [CHANGELOG][] из [GitHub-репозитария][gh], я
  специально избегал цели создать *стандарт* или чёткий список правил (как это
  делает, например, [SemVer.org][semver]).

  Я сделал это потому, что хочу, чтобы наше сообщество достигло консенсуса. Я
  верю, что дискуссия также важна, как и её результат.

  Так что, пожалуйста, [**участвуйте**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/ru/1.0.0/index.html.haml
```haml
---
title: Ведите changelog
description: Не позволяйте своим друзьям сливать логи Git в changelog’и.
language: ru
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Что такое лог изменений?

  %p
    Лог изменений — это файл, который содержит поддерживаемый, хронологически
    упорядоченный список значимых изменений для каждой версии проекта.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Зачем вести лог изменений?

  %p
    Чтобы пользователям и контрибуторам было проще в точности понять,
    какие значимые изменения были внесены в каждый выпуск (или версию)
    проекта.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Кому нужен лог изменений?

  %p
    Людям. Конечные пользователи программного обеспечения, будь то клиенты или разработчики, —
    это человеческие существа, которым небезразлично, с чем они работают. Когда
    программное обеспечение изменяется, люди хотят знать, что и почему изменилось.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Как мне сделать хороший лог изменений?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Руководящие принципы

  %ul
    %li
      Лог изменений — <em>для людей</em>, а не для машин.
    %li
      Для каждой версии без исключения следует создать отдельный раздел.
    %li
      Однотипные изменения следует группировать.
    %li
      Следует предусмотреть возможность поставить ссылку на любую версию или раздел.
    %li
      Последняя версия должна идти в начале файла.
    %li
      Указаны даты выпуска каждой версии.
    %li
      Уточните, следуете ли вы принципам #{link_to "семантического версионирования", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Типы изменений

  %ul
    %li
      %code Добавлено
      — для новых функций.
    %li
      %code Изменено
      — для изменений в существующей функциональности.
    %li
      %code Устарело
      — для функций, которые скоро будут удалены.
    %li
      %code Удалено
      — для удалённых на данный момент функций.
    %li
      %code Исправлено
      — для любых исправлений багов.
    %li
      %code Безопасность
      — на случай уязвимостей.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Как мне тратить меньше усилий на ведение лога изменений?

  %p
    Держите в начале файла раздел <code>Новое</code>, позволяющий отслеживать
    предстоящие изменения.

  %p Это служит достижению двух целей:

  %ul
    %li
      люди смогут видеть, каких изменений им можно ожидать в предстоящих выпусках;
    %li
      в момент релиза вы можете переместить изменения из раздела <code>Новое</code>
      в раздел нового выпуска.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Бывают ли плохие логи изменений?

  %p Да. Вот несколько причин, по которым они могут оказаться совершенно бесполезными.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diff’ы лога коммитов

  %p
    Использование diff’ов лога коммитов в качестве лога изменений — это плохая идея:
    они полны информационного шума от слияния коммитов, от коммитов с непонятными
    заглавиями, от изменений, вносимых в документацию, и т. п.

  %p
    Назначение коммита в том, чтобы задокументировать шаг в эволюции исходного
    кода. В некоторых проектах следят за историей коммитов, в некоторых — нет.

  %p
    Назначение же раздела в логе изменений — задокументировать заслуживающие
    внимания различия (зачастую привнесённые несколькими коммитами), чтобы внятно
    сообщить конечным пользователям об этих различиях.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Игнорирование устаревших функций

  %p
    Когда люди переходят с одной версии продукта на другую, им должно быть до боли
    ясно, в какой именно момент что-то сломается. Следует предусмотреть возможность
    перейти к версии, в которой перечислены устаревшие функции, удалить то,
    что устарело, а затем перейти к версии, из которой эти устаревшие функции
    удалены.

  %p
    Перечисляйте в логе изменений устаревшие и удалённые функции, а также любые
    критические изменения, даже если не перечисляете ничего другого.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Даты, сбивающие с толку

  %p
    Региональные форматы дат различаются по всему миру, и зачастую трудно найти
    удобный для человека формат, который был бы интуитивно понятен каждому.
    Преимущество дат в форматах наподобие <code>2017-07-17</code> заключается
    в том, что элементы в них следуют в порядке от более крупной единицы измерения
    к более мелкой: год, месяц и день. К тому же, в отличие от некоторых
    региональных форматов, в которых изменено положение чисел, обозначающих день
    и месяц, этот формат не пересекается с другим и не вызывает неоднозначных
    толкований. Исходя из этих причин и того факта, что этот формат соответствует
    #{link_to "стандарту ISO", data.links.isodate}, именно он рекомендован для записей в логе
    изменений.

  %aside
    Есть кое-что ещё. Помогите мне собрать эти антипаттерны,
    подав #{link_to "заявку о наличии проблемы", data.links.issue}
    или pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Часто задаваемые вопросы

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Существует ли стандартный формат для логов изменений?

  %p
    На самом деле, нет. Есть #{link_to "стилистический путеводитель по логам изменений от GNU", data.links.gnustyle},
    есть #{link_to "«руководство» длиной в два абзаца по файлам GNU NEWS", data.links.gnunews}.
    Оба или неадекватны, или недостаточно полны.

  %p
    Этот проект нацелен на то, чтобы стать
    #{link_to "улучшенной версией соглашения о формате логов изменений", data.links.changelog}.
    Проект опирается на отслеживание и накопление передового опыта сообщества
    пользователей открытого исходного кода.

  %p
    Здоровая критика, дискуссии и предложения по улучшению
    #{link_to "приветствуются", data.links.issue}.


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Как назвать файл лога изменений?

  %p
    Назовите его <code>CHANGELOG.md</code>. Некоторые проекты используют
    <code>HISTORY</code>, <code>NEWS</code> или <code>RELEASES</code>.

  %p
    Хотя легко подумать, что имя файла лога изменений не имеет большого значения,
    зачем усложнять вашим конечным пользователям поиск места,
    в котором описаны все значимые изменения?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Что насчёт функции «Релизы» на GitHub’е?

  %p
    Это отличная инициатива. #{link_to "Релизы", data.links.github_releases} можно использовать
    для превращения простых тегов Git (например, тега, названного <code>v1.0.0</code>)
    в подробные примечания к выпускам, вручную добавляя эти примечания, или же
    можно извлечь сообщения из аннотированных тегов Git и превратить их в примечания.

  %p
    Релизы на GitHub’е создают непортируемый лог изменений, который может быть
    показан пользователям только на самом GitHub’е. Имеется возможность вести такой лог
    в формате, очень похожем на формат проекта Keep a Changelog, но это, как правило,
    требует большей вовлечённости в процесс.

  %p
    Также возможно, что конечным пользователям не всегда легко обнаружить
    текущую версию GitHub Releases, в отличие от обычных файлов с именами в верхнем
    регистре (<code>README</code>, <code>CONTRIBUTING</code> и т. д.).
    Другая небольшая проблема заключается в том, что в настоящее время
    интерфейс не предлагает ссылок на логи коммитов, выполненных между релизами.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Могут ли логи изменений быть автоматически распарсены?

  %p
    Это сложно, потому что люди соблюдают сильно различающиеся форматы
    и используют разные имена файлов.

  %p
    #{link_to "Vandamme", data.links.vandamme} — это gem для Ruby, созданный командой
    Gemnasium и способный парсить логи изменений во многих (но не всех)
    проектах с открытым исходным кодом.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Что насчёт yanked-выпусков?

  %p
    Yanked-выпуски — это версии, которые пришлось изъять из обращения из-за
    серьёзного бага или проблем с безопасностью. Часто такие версии даже
    не обозначают в логах изменений. А следовало бы. И вот как вам следует их
    обозначать:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Тег <code>[YANKED]</code> так бросается в глаза неспроста. Очень важно,
    чтобы люди его заметили. Поскольку он заключён в квадратные скобки,
    его также проще распарсить программно.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Следует ли вам когда-либо переписывать лог изменений?

  %p
    Конечно. Всегда есть веские причины для усовершенствования лога изменений.
    Я регулярно подаю pull request’ы на добавление недостающих выпусков в проекты
    с открытым исходным кодом, которые оставили свои логи изменений без сопровождения.

  %p
    К тому же, возможно, вы обнаружите, что в примечании к версии
    забыли рассмотреть одно из критичных изменений. Важность того,
    что в этом случае вы обновите ваш лог изменений, очевидна.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Как я могу помочь вашему проекту?

  %p
    Этот документ — <strong>не истина в последней инстанции</strong>; это моё
    тщательно обдуманное мнение наряду с информацией и примерами, которые я собрал.

  %p
    Дело в том, что я хочу, чтобы наше сообщество пришло к согласованному мнению.
    Я верю, что дискуссия так же важна, как и конечный результат.

  %p
    Так что, пожалуйста, <strong>#{link_to "участвуйте", data.links.repo}</strong>.

.press
  %h3 Обсуждения
  %p
    Я приходил на #{link_to "подкаст The Changelog", data.links.thechangelog}, чтобы поговорить о том,
    почему maintainer’ам (персоналу сопровождения) и контрибуторам следует озаботиться
    ведением логов изменений, а также о мотивах, стоящих за этим проектом.
```

## File: source/ru/1.1.0/index.html.haml
```haml
---
title: Ведите changelog
description: Не позволяйте своим друзьям сливать логи Git в changelog’и.
language: ru
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Что такое лог изменений?

  %p
    Лог изменений — это файл, который содержит поддерживаемый, хронологически
    упорядоченный список значимых изменений для каждой версии проекта.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Зачем вести лог изменений?

  %p
    Чтобы пользователям и контрибуторам было проще в точности понять,
    какие значимые изменения были внесены в каждый выпуск (или версию)
    проекта.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Кому нужен лог изменений?

  %p
    Людям. Конечные пользователи программного обеспечения, будь то клиенты или разработчики, —
    это человеческие существа, которым небезразлично, с чем они работают. Когда
    программное обеспечение изменяется, люди хотят знать, что и почему изменилось.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Как мне сделать хороший лог изменений?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Руководящие принципы

  %ul
    %li
      Лог изменений — <em>для людей</em>, а не для машин.
    %li
      Для каждой версии без исключения следует создать отдельный раздел.
    %li
      Однотипные изменения следует группировать.
    %li
      Следует предусмотреть возможность поставить ссылку на любую версию или раздел.
    %li
      Последняя версия должна идти в начале файла.
    %li
      Указаны даты выпуска каждой версии.
    %li
      Уточните, следуете ли вы принципам #{link_to "семантического версионирования", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Типы изменений

  %ul
    %li
      %code Добавлено
      — для новых функций.
    %li
      %code Изменено
      — для изменений в существующей функциональности.
    %li
      %code Устарело
      — для функций, которые скоро будут удалены.
    %li
      %code Удалено
      — для удалённых на данный момент функций.
    %li
      %code Исправлено
      — для любых исправлений багов.
    %li
      %code Безопасность
      — на случай уязвимостей.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Как мне тратить меньше усилий на ведение лога изменений?

  %p
    Держите в начале файла раздел <code>Новое</code>, позволяющий отслеживать
    предстоящие изменения.

  %p Это служит достижению двух целей:

  %ul
    %li
      люди смогут видеть, каких изменений им можно ожидать в предстоящих выпусках;
    %li
      в момент релиза вы можете переместить изменения из раздела <code>Новое</code>
      в раздел нового выпуска.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Бывают ли плохие логи изменений?

  %p Да. Вот несколько причин, по которым они могут оказаться совершенно бесполезными.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diff’ы лога коммитов

  %p
    Использование diff’ов лога коммитов в качестве лога изменений — это плохая идея:
    они полны информационного шума от слияния коммитов, от коммитов с непонятными
    заглавиями, от изменений, вносимых в документацию, и т. п.

  %p
    Назначение коммита в том, чтобы задокументировать шаг в эволюции исходного
    кода. В некоторых проектах следят за историей коммитов, в некоторых — нет.

  %p
    Назначение же раздела в логе изменений — задокументировать заслуживающие
    внимания различия (зачастую привнесённые несколькими коммитами), чтобы внятно
    сообщить конечным пользователям об этих различиях.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Игнорирование устаревших функций

  %p
    Когда люди переходят с одной версии продукта на другую, им должно быть до боли
    ясно, в какой именно момент что-то сломается. Следует предусмотреть возможность
    перейти к версии, в которой перечислены устаревшие функции, удалить то,
    что устарело, а затем перейти к версии, из которой эти устаревшие функции
    удалены.

  %p
    Перечисляйте в логе изменений устаревшие и удалённые функции, а также любые
    критические изменения, даже если не перечисляете ничего другого.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Даты, сбивающие с толку

  %p
    Региональные форматы дат различаются по всему миру, и зачастую трудно найти
    удобный для человека формат, который был бы интуитивно понятен каждому.
    Преимущество дат в форматах наподобие <code>2017-07-17</code> заключается
    в том, что элементы в них следуют в порядке от более крупной единицы измерения
    к более мелкой: год, месяц и день. К тому же, в отличие от некоторых
    региональных форматов, в которых изменено положение чисел, обозначающих день
    и месяц, этот формат не пересекается с другим и не вызывает неоднозначных
    толкований. Исходя из этих причин и того факта, что этот формат соответствует
    #{link_to "стандарту ISO", data.links.isodate}, именно он рекомендован для записей в логе
    изменений.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Непоследовательное освещение изменений

  %p
    Лог, в котором упомянуты лишь некоторые изменения, может быть опасен в той же мере,
    что и отсутствие лога изменений. Несмотря на то, что многие изменения могут быть
    нерелевантными (например, удаление одного пробела не во всех случаях может нуждаться
    в регистрации), в логе следует упоминать о любых важных изменениях.
    При непоследовательном освещении изменений ваши пользователи будут заблуждаться,
    считая лог изменений единственным источником истины. А ведь таким он и должен быть.
    С большой силой приходит большая ответственность — наличие хорошего лога изменений
    означает наличие последовательно обновляемого лога изменений.

  %aside
    Есть кое-что ещё. Помогите мне собрать эти антипаттерны,
    подав #{link_to "заявку о наличии проблемы", data.links.issue}
    или pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Часто задаваемые вопросы

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Существует ли стандартный формат для логов изменений?

  %p
    На самом деле, нет. Есть #{link_to "стилистический путеводитель по логам изменений от GNU", data.links.gnustyle},
    есть #{link_to "«руководство» длиной в два абзаца по файлам GNU NEWS", data.links.gnunews}.
    Оба или неадекватны, или недостаточно полны.

  %p
    Этот проект нацелен на то, чтобы стать
    #{link_to "улучшенной версией соглашения о формате логов изменений", data.links.changelog}.
    Проект опирается на отслеживание и накопление передового опыта сообщества
    пользователей открытого исходного кода.

  %p
    Здоровая критика, дискуссии и предложения по улучшению
    #{link_to "приветствуются", data.links.issue}.


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Как назвать файл лога изменений?

  %p
    Назовите его <code>CHANGELOG.md</code>. Некоторые проекты используют
    <code>HISTORY</code>, <code>NEWS</code> или <code>RELEASES</code>.

  %p
    Хотя легко подумать, что имя файла лога изменений не имеет большого значения,
    зачем усложнять вашим конечным пользователям поиск места,
    в котором описаны все значимые изменения?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Что насчёт функции «Релизы» на GitHub’е?

  %p
    Это отличная инициатива. #{link_to "Релизы", data.links.github_releases} можно использовать
    для превращения простых тегов Git (например, тега, названного <code>v1.0.0</code>)
    в подробные примечания к выпускам, вручную добавляя эти примечания, или же
    можно извлечь сообщения из аннотированных тегов Git и превратить их в примечания.

  %p
    Релизы на GitHub’е создают непортируемый лог изменений, который может быть
    показан пользователям только на самом GitHub’е. Имеется возможность вести такой лог
    в формате, очень похожем на формат проекта Keep a Changelog, но это, как правило,
    требует большей вовлечённости в процесс.

  %p
    Также возможно, что конечным пользователям не всегда легко обнаружить
    текущую версию GitHub Releases, в отличие от обычных файлов с именами в верхнем
    регистре (<code>README</code>, <code>CONTRIBUTING</code> и т. д.).
    Другая небольшая проблема заключается в том, что в настоящее время
    интерфейс не предлагает ссылок на логи коммитов, выполненных между релизами.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Могут ли логи изменений быть автоматически распарсены?

  %p
    Это сложно, потому что люди соблюдают сильно различающиеся форматы
    и используют разные имена файлов.

  %p
    #{link_to "Vandamme", data.links.vandamme} — это gem для Ruby, созданный командой
    Gemnasium и способный парсить логи изменений во многих (но не всех)
    проектах с открытым исходным кодом.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Что насчёт yanked-выпусков?

  %p
    Yanked-выпуски — это версии, которые пришлось изъять из обращения из-за
    серьёзного бага или проблем с безопасностью. Часто такие версии даже
    не обозначают в логах изменений. А следовало бы. И вот как вам следует их
    обозначать:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Тег <code>[YANKED]</code> так бросается в глаза неспроста. Очень важно,
    чтобы люди его заметили. Поскольку он заключён в квадратные скобки,
    его также проще распарсить программно.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Следует ли вам когда-либо переписывать лог изменений?

  %p
    Конечно. Всегда есть веские причины для усовершенствования лога изменений.
    Я регулярно подаю pull request’ы на добавление недостающих выпусков в проекты
    с открытым исходным кодом, которые оставили свои логи изменений без сопровождения.

  %p
    К тому же, возможно, вы обнаружите, что в примечании к версии
    забыли рассмотреть одно из критичных изменений. Важность того,
    что в этом случае вы обновите ваш лог изменений, очевидна.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Как я могу помочь вашему проекту?

  %p
    Этот документ — <strong>не истина в последней инстанции</strong>; это моё
    тщательно обдуманное мнение наряду с информацией и примерами, которые я собрал.

  %p
    Дело в том, что я хочу, чтобы наше сообщество пришло к согласованному мнению.
    Я верю, что дискуссия так же важна, как и конечный результат.

  %p
    Так что, пожалуйста, <strong>#{link_to "участвуйте", data.links.repo}</strong>.

.press
  %h3 Обсуждения
  %p
    Я приходил на #{link_to "подкаст The Changelog", data.links.thechangelog}, чтобы поговорить о том,
    почему maintainer’ам (персоналу сопровождения) и контрибуторам следует озаботиться
    ведением логов изменений, а также о мотивах, стоящих за этим проектом.
```

## File: source/sk/1.0.0/index.html.haml
```haml
---
title: Udržuj changelog
description: Nenechaj kamarátov sypať git logy do changelogov.
language: sk
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Verzia
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Čo je to changelog?

  %p
    Changelog je súbor obsahujúci organizovaný, chronologicky usporiadaný
    zoznam významných zmien pre každú verziu daného projektu.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Prečo udržiavať changelog?

  %p
    Aby používatelia a prispievatelia presne vedeli, aké podstatné zmeny
    sa udiali medzi jednotlivými vydaniami (alebo verziami) projektu.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kto potrebuje changelog?

  %p
    Ľudia. Či už konzumenti, alebo vývojári, koncoví používatelia softvéru
    sú ľudské bytosti, ktorým záleží na tom, čo softvér obsahuje. Keď sa
    softvér zmení, ľudia chcú vedieť prečo a ako.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Ako vytvorím dobrý changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Hlavné princípy

  %ul
    %li
      Changelogy sú <em>pre ľudí</em>, nie stroje.
    %li
      Changelog by mal obsahovať záznam pre každú jednu verziu.
    %li
      Rovnaké typy zmien by mali byť zoskupené.
    %li
      Mala by existovať možnosť odkazovať na jednotlivé verzie a sekcie.
    %li
      Posledná verzia je uvedená na prvom mieste.
    %li
      Pre každú verziu je uvedený dátum jej vydania.
    %li
      Uveď tiež, že sa držíš #{link_to "sémantického verziovania", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Typy zmien

  %ul
    %li
      %code Added
      pre nové funkcie.
    %li
      %code Changed
      pre zmeny existujúcej funkcie.
    %li
      %code Deprecated
      pre funkcie, ktoré budú čoskoro odstránené.
    %li
      %code Removed
      pre odstránené funkcie.
    %li
      %code Fixed
      pre opravy chýb.
    %li
      %code Security
      v prípade bezpečnostných zraniteľností.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Ako minimalizovať úsilie vynaložené na udržiavanie changelogu?

  %p
    Udržuj <code>Unreleased</code> sekciu na začiatku changelogu
    pre nadchádzajúce zmeny.

  %p Má to dva dôvody:

  %ul
    %li
      Ľudia môžu vidieť, aké zmeny môžu očakávať v ďalších vydaniach
    %li
      V čase vydávania novej verzie môžeš presunúť zmeny z
      <code>Unreleased</code> sekcie do sekcie novej verzie

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Môžu byť changelogy zlé?

  %p Áno. Tu je hneď niekoľko spôsobov, ako môžu byť neužitočné.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffy z commit logu

  %p
    Použitie diffov z commit logu ako changelog nie je dobrý nápad:
    sú plné balastu. Veci ako merge commity, commity s nejasnými názvami,
    zmeny dokumentácie a pod.

  %p
    Účel commitu je dokumentovanie kroku v evolúcii zdrojového kódu.
    Niektoré projekty commity prečisťujú, iné nie.

  %p
    Účelom changelogu je dokumentovanie významných zmien, často naprieč
    viacerými commitmi, a jasne ich komunikovať koncovému používateľovi.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorovanie oznámenia zastaralých funkcií

  %p
    Keď používatelia prechádzajú na novšiu verziu, musí byť jasné, čo sa
    rozbije. Mala by pre nich existovať možnosť prejsť na verziu so zoznamom
    zastaralých funkcií, tieto funkcie odstrániť a následne prejsť na verziu,
    kde sú tieto zastaralé funkcie už odstránené.

  %p
    Ak už nič iné, tak aspoň uveď zastaralé, odstránené funkcie a všetky zmeny,
    ktoré môžu niečo rozbiť, do changelogu.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Mätúce dátumy

  %p
    Regionálne formáty dátumov sa líšia naprieč svetom a často je zložité
    nájsť formát dátumu, ktorý by bol prívetivý a intuitívny pre všetkých
    používateľov. Výhodou dátumu vo formáte <code>2017-07-17</code> je poradie
    od najväčšej jednotky po najmenšiu: rok, mesiac, deň. Tento formát sa tiež
    neprekrýva s inými formátmi, ktoré zamieňajú pozíciu dňa a mesiaca. Kvôli
    týmto dôvodom spolu s faktom, že ide o #{link_to "ISO štandard", data.links.isodate},
    je tento formát odporučený pre záznamy v changelogu.

  %aside
    Je toho však viac. Pomôž mi zozbierať tieto antivzory
    = link_to "otvorením issue", data.links.issue
    alebo pull requestom.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Často kladené otázky

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Existuje štandardný formát pre changelog?

  %p
    Nie. Existuje GNU príručka pre štýl changelogu alebo dvojodstavcová
    GNU "smernica" pre NEWS súbor. Obe sú však nevhodné či nedostatočné.

  %p
    Tento projekt sa snaží byť
    = link_to "lepšou konvenciou pre changelog.", data.links.changelog
    Vychádza z pozorovania a zozbierania osvedčených postupov komunity okolo projektov s otvoreným zdrojovým kódom.

  %p
    Zdravá kritika, diskusia a návrhy na zlepšenie
    = link_to "sú vítané.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Ako by mal byť súbor changelogu pomenovaný?

  %p
    Nazvi ho <code>CHANGELOG.md</code>. Niektoré projekty používajú tiež
    <code>HISTORY</code>, <code>NEWS</code> alebo <code>RELEASES</code>.

  %p
    Je jednoduché myslieť si, že názov changelogu nie je taký dôležitý.
    Prečo však sťažovať koncovému používateľovi hľadanie významných zmien?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    A čo GitHub Releases?

  %p
    Je to skvelá iniciatíva. Služba #{link_to "Releases", data.links.github_releases} môže byť použitá
    pre premenu git tagov (napríklad tagu <code>v1.0.0</code>) na bohaté
    poznámky k vydaniam manuálnym pridávaním týchto poznámok alebo získaním
    správ z anotovaných git tagov a vytvorením poznámok k vydaniu z nich.

  %p
    GitHub Releases vytvorí neprenosný changelog, ktorý môže byť zobrazený
    používateľom v rámci GitHubu. Je možné ich upraviť na veľmi podobný štýl,
    aký navrhuje projekt Udržuj changelog, tento postup však býva trochu
    zložitejší.

  %p
    Súčasná verzia GitHub Releases tiež nie je ľahko objaviteľná koncovým
    používateľom, na rozdiel od klasického súboru s názvom napísaným veľkými
    písmenami (<code>README</code>, <code>CONTRIBUTING</code> atď.). Ďalším
    menším problémom je, že v súčasnosti nepodporuje odkazy na commit logy
    medzi jednotlivými vydaniami.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Môžu byť changelogy automaticky parsované?

  %p
    Je to zložité, pretože ľudia používajú rôzne formáty a názvy súborov.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Ruby gem vytvorený tímom
    Gemnasium, ktorý parsuje mnohé (ale nie všetky)
    changelogy projektov s otvoreným zdrojovým kódom.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    A čo spätne stiahnuté vydania?

  %p
    Stiahnuté vydania sú verzie, ktoré museli byť neskôr spätne odobraté
    kvôli vážnej chybe alebo bezpečnostným rizikám. Tieto verzie sa často
    v changelogu ani neobjavia. Ale mali by sa. Zobrazovať by sa mali takto:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Tag <code>[YANKED]</code> kričí naschvál. Je dôležité, aby si ho ľudia
    všimli. Keďže je uzavretý zátvorkami, je tiež jednoduchšie ho programovo
    parsovať.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Mal by sa changelog niekedy prepisovať?

  %p
    Samozrejme. Vždy sa nájde dobrý dôvod na vylepšenie changelogu. Sám často
    otváram pull requesty pre pridanie chýbajúceho vydania projektov
    s otvoreným kódom a neudržiavaným changelogom.

  %p
    Tiež môže nastať situácia, že v poznámkach k vydaniu určitej verzie sa
    zabudla spomenúť podstatná zmena. V takom prípade je samozrejme dôležité
    tento changelog aktualizovať.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Ako môžem prispieť?

  %p
    Tento dokument nie je úplná <strong>pravda</strong>; je mojím starostlivo
    zváženým názorom spolu s informáciami a príkladmi, ktoré som zozbieral.

  %p
    Je tomu tak preto, aby komunita dosiahla určitý konsenzus. Verím, že
    diskusia je rovnako dôležitá ako samotný výsledok.

  %p
    Takže, prosím, <strong>#{link_to "prilož ruku k dielu", data.links.repo}</strong>.

.press
  %h3 Rozhovory
  %p
    Zúčastnil som sa podcastu #{link_to "The Changelog", data.links.thechangelog},
    kde sme sa rozprávali o tom, prečo by sa správci projektov a prispievatelia
    mali zaujímať o changelogy a tiež o motivácii za týmto projektom.
```

## File: source/sl/0.3.0/index.html.haml
```haml
---
title: Keep a Changelog
description: Ne dopustite, da vaši prijatelji odlagajo git dnevnike v CHANGELOG™
language: sl
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Kaj je dnevnik sprememb?
  Dnevnik sprememb je datoteka, ki vsebuje kuriran, kronološko urejen
  seznam opaznih sprememb za vsako verzijo projekta.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Kaj je smisel dnevnika sprememb?
  Za poenostavitev, da uporabniki in prispevajoči sodelavci natančno vidijo, katere
  opazne spremembe so bile narejene med vsako izdajo (ali verzijo) projekta.

  ### Zakaj se truditi?
  Ker so programska orodja za ljudi. Če vam to ni pomembno, zakaj
  prispevate odprto kodnemu projektu? Zagotovo mora obstajati malenkost (hehe)
  skrbnosti nekje v vaši lepi glavi.

  Govorili smo z [Adam-om Stacoviak-om in Jerod-om Santo na temo Changelog-a][thechangelog]
  (ustrezno, kajne?) podcast o tem zakaj bi se morali vzdrževalci in
  sodelavci, ki prispevajo, truditi in o motivaciji za tem projektom.
  Če imate na voljo nekaj časa (1:06), je odlično poslušanje.

  ### Kaj naredi dnevnik sprememb odličen?
  Veseli me, da ste vprašali.

  Dober dnevnik sprememb se drži teh načel:

  - Je enostaven za ljudi, ne stroje, tako da je čitljivost ključnega pomena.
  - Enostaven za povezavo kateri koli sekciji (torej Markdown pred golim besedilom).
  - Ena pod-sekcija na verzijo.
  - Seznam izdaj v obratnem kronološkem vrstnem redu (najnovejše na vrhu).
  - Zapis vseh datumov v `YYYY-MM-DD` formatu. (Na primer: `2012-06-02` za `2. junij 2012`.) Je mednarnodno, [smiselno](https://xkcd.com/1179/) in neodvisno od jezika.
  - Eksplicitna omemba ali projekt sledi [semantičnim verzijam][semver].
  - Vsaka verzija bi morala:
    - Imeti seznam njenih datumov izdaje v zgornjem formatu.
    - Skupine sprememb, ki opisujejo njihove vplive na projekt, kot sledi:
      - `Added` za nove lastnosti.
      - `Changed` za spremembe obstoječih lastnosti.
      - `Deprecated` za nekoč stabilne lastnosti, ki bodo odstranjene v prihajajočih verzijah.
      - `Removed` za zastarele lastnosti, ki so odstranjene v tej izdaji.
      - `Fixed` za kakršne koli popravke hroščev.
      - `Security` za povabilo uporabnikom, da nadgradijo v primeru varnostnih ranljivosti.

  ### Kako zmanjšati potreben trud?
  Vedno imejte `"Unreleased"` sekcijo na vrhu za sledenje katerih koli
  sprememb.

  To služi dvema namenoma:

  - Ljudje lahko vidijo, katere spremembe lahko pričakujejo v prihajajočih izdajah
  - Pri izdaji, morate tako samo spremembiti `"Unreleased"` v številko verzije
    in dodati nov naslov `"Unreleased"` na vrh.

  ### Kaj spravi samoroga v jok?
  Dobro … pojdimo skozi.

  - **Odlaganje sprememb git dnevnikov poslanih sprememb.** Temu se izogibajte, saj s tem ne pomagate nikomur.
  - **Nepoudarjanje zastarelosti.** Ko uporabniki nadgradijo iz ene verzije na
    drugo, bi moralo biti enostavno jasno, ko nekaj ne bo več delovalo.
  - **Datumi v lokalnih oblikah.** V ZDA, uporabniki dajo mesec na prvo mesto
    ("06-02-2012" za 2. junij, 2012, kar je brez smisla), medtem ko mnogi ostali
    uporabniki drugod po svetu pišejo "2 June 2012" robotskega izgleda, vendar
    izgovorijo to drugače. "2012-06-02" deluje logično po večini in se ne
    prekriva na dvoumne načine z ostalimi oblikami datumov. To je tudi
    [ISO standard](http://www.iso.org/iso/home/standards/iso8601.htm). Tako je to
    priporočljiva oblika datuma za dnevnike sprememb.

  Tega je še več. Pomagajte zbrati te solze samoroga preko
  [odprtja težave][issues]
  ali zahtevka potega.

  ### Ali obstaja standardna oblika dnevnika sprememb?
  Na žalost ne. Vendar mirno kri. Vem, da besno hitite najti to povezavo
  v vodiču dnevnika sprememb GNU ali v datoteki dveh odstavkov "guideline" GNU
  NEWS. Stilski vodič GNU je lep začetek, vendar je na žalost preveč enostaven.
  S tem ni nič narobe, vendar ko uporabniki potrebujejo
  vodič, je redko v pomoč. Posebej, ko je za reševati veliko
  situacij in posebnih primerov.

  Ta projekt [vsebuje, kar upamo, da bo postalo boljša datoteka konvencij CHANGELOG][CHANGELOG].
  Mislimo, da status quo ni dovolj dober in mislimo, da lahko kot skupnost
  naredimo boljšo konvencijo, če poskusimo izvleči dobre prakse iz
  realnih programskih projektov. Prosimo, poglejte naokoli in si zapomnite, da
  [diskusije in predlogi za izboljšave so dobrodošli][issues]!

  ### Kako se mora dnevnik sprememb imenovati?
  Če niste uspeli razbrati iz zgornjega primera, je `CHANGELOG.md`
  najboljša konvencija do sedaj.

  Nekateri projekti uporabljajo tudi `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md` itd.

  Gre za zmešnjavo. Vsa ta imena ljudem samo otežijo najti dnevnik sprememb.

  ### Zakaj uporabniki ne morejo uporabiti enostavno samo `git log` sprememb?
  Ker so spremembe dnevnika polne šuma - privzeto. Lahko bi naredili ustrezen
  dnevnik sprememb tudi v hipotetičnem projektu, ki ga poganjajo popolni ljudje, ki nikoli ne naredijo
  napak, nikoli ne pozabijo poslati novih datotek in nikoli ne zamudijo nobenega dela refaktorizacije.
  Razlog pošiljke (commit-a) je dokumentirati en atomičen korak v procesu preko katerega
  se koda razvija iz enega stanja v drugo. Razlog dnevnika sprememb je
  dokumentiranje omembe vrednih sprememb med temi stanji.

  Kakor obstaja razlika med dobrimi komentarji in samo kodo,
  tako je razlika med dnevnikom sprememb in dnevnikom git commit-a:
  eden opisuje *zakaj*, drugi pa *kako*.

  ### Ali se da dnevnik sprememb samodejno razlčeniti?
  Je težko, ker uporabniki sledijo divje različnim formatom in imenom datotek.

  [Vandamme][vandamme] je Ruby gem,
  ki ga je ustvarila ekipa [Gemnasium][gemnasium]. Razčlenjuje
  mnoge (vendar ne vse) dnevnike sprememb odprto kodnih projektov.

  ### Zakaj razlikovanje med črkovanjem "CHANGELOG" in "change log"?
  "CHANGELOG" je ime same datoteke. Je nekoliko glasno, vendar je to
  zgodovinska konvencija, kateri sledi mnogo odprto kodnih projektov. Drugi
  primeri podobnih datotek vključujejo [`README`][README], [`LICENSE`][LICENSE],
  in [`CONTRIBUTING`][CONTRIBUTING].

  Ime z velikimi črkami (kar je datoteko v starih operacijskih sistemih prikazalo na
  vrhu) je uporabljeno, da se povleče pozornost nanje. Ker gre za pomembne
  meta podatke o projektu, so lahko uporabne za kogarkoli, ki namerava uporabiti
  ali prispevati, precej enako kot [značke odprto kodnih projektov][shields].

  Ko se sklicujemo na "change log", govorimo o funkciji te
  datoteke: beleženje sprememb.

  ### Kaj pa povlečene izdaje?
  T.i. yanked ali povlečene izdaje so verzije, ki so bile primorane biti potegnjene zaradi resnega
  hrošča ali varnostne težave. Pogostokrat se te verzije niti ne pojavijo v
  dnevnikih sprememb. Vendar bi se morale. Prikazane bi morale biti sledeče:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Oznaka `[YANKED]` je izpisana z velikimi črkami z razlogom. Pomembno je, da jo uporabniki
  opazijo. Ker je obdana z oglatimi oklepaji, jo je tudi enostavnejše razčleniti
  programsko.

  ### Ali bi morali kadarkoli prepisati dnevnik sprememb?
  Seveda. Vedno obstajajo dobri razlogi, kako izboljšati dnevnik sprememb. Odpiranje
  zahtevkov potegov je pogostokrat priporočljivo, da se doda manjkajoče izdaje
  odprtokodnih projektov z nevzdrževanimi dnevniki sprememb.

  Možno je tudi, da odkrijete, da ste pozabili nasloviti pomembne spremembe
  v opombah verzije. Očitno je pomembno, da vaš
  dnevnik sprememb v tem primeru posodobite.

  ### Kako lahko pomagam?
  Ta dokument ni **dejstvo**; je osebno temeljito premišljeno
  mnenje, skupaj z informacijami in primeri, ki smo jih zbrali.
  Čeprav ponujamo dejanski [CHANGELOG][] v [GitHub repozitoriju][gh],
  namenoma ni ustvarjene primerne *izdaje* ali jasnega seznama pravil
  za sledenje (kot ima to npr. [SemVer.org][semver]).

  To je zato, ker želimo, da naša skupnost doseže soglasje. Verjamemo,
  da je razprava tako pomembna kot končni rezultat.

  Torej, prosimo [**pridružite se**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/sl/1.0.0/index.html.haml
```haml
---
title: Vzdržujte dnevnik sprememb
description: Ne dopustite, da vaši prijatelji odlagajo dnevnike Git v dnevnike sprememb.
language: sl
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Različica
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Kaj je dnevnik sprememb?

  %p
    Dnevnik sprememb (angl. changelog) je datoteka, ki vsebuje kuriran, kronološko
    urejen seznam opaznih sprememb za vsako različico projekta.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Zakaj vzdrževati dnevnik sprememb?

  %p
    Za poenostavitev, da uporabniki in sodelavci natančno vidijo, katere
    opazne spremembe so bile narejene med vsako izdajo (ali različico)
    projekta.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kdo potrebuje dnevnik sprememb?

  %p
    Ljudje to potrebujejo. Ne glede na to, ali so uporabniki ali razvijalci, so končni uporabniki
    programske opreme ljudje, ki jim je mar, kaj je v programski opremi. Ko se
    programska oprema spremeni, ljudje želijo vedeti, zakaj in kako.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Kako naredim dober dnevnik sprememb?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Smernice

  %ul
    %li
      Dnevniki sprememb so <em>za ljudi</em>, ne za stroje.
    %li
      Obstajati mora vnos za vsako posamezno različico.
    %li
      Enake vrste sprememb je treba združiti.
    %li
      Različice in razdelki morajo biti povezljivi.
    %li
      Najnovejša različica je na prvem mestu.
    %li
      Prikaže se datum izdaje vsake različice.
    %li
      Navedite, ali sledite #{link_to "Semantičnim različicam", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Vrste sprememb

  %ul
    %li
      %code Dodano (angl. Added)
      za nove lastnosti.
    %li
      %code Spremenjeno (angl. Changed)
      za spremembe obstoječe funkcionalnosti.
    %li
      %code Opuščeno (angl. Deprecated)
      za lastnosti, ki bodo kmalu odstranjene.
    %li
      %code Odstranjeno (angl. Removed)
      za zdaj odstranjene lastnosti.
    %li
      %code Popravljeno (angl. Fixed)
      za morebitne popravke napak.
    %li
      %code Varnost (angl. Security)
      v primeru ranljivosti.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Kako lahko zmanjšam trud, potreben za vzdrževanje dnevnika sprememb?

  %p
    Na vrhu imejte razdelek <code>Neobjavljeno</code>, če želite slediti prihajajočim
    spremembam.

  %p To ima dva namena:

  %ul
    %li
      Ljudje lahko vidijo, kakšne spremembe lahko pričakujejo v prihajajočih izdajah
    %li
      Ob izdaji lahko premaknete razdelek sprememb <code>Neobjavljeno</code>
      v nov razdelek različice izdaje.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Ali so lahko dnevniki sprememb slabi?

  %p Da. Tukaj je nekaj načinov, kako so lahko manj koristni.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Razlike dnevnika potrditev (angl. diffs)

  %p
    Uporaba razlik dnevnika potrditev kot dnevnikov sprememb je slaba ideja: polni so
    navlake. Stvari, kot so potrditve združitev, potrditve z nejasnimi naslovi,
    spremembe dokumentacije itd.

  %p
    Namen potrditve je dokumentirati korak v razvoju
    izvorne kode. Nekateri projekti imajo potrditve čiste, nekateri ne.

  %p
    Namen vnosa v dnevnik sprememb je dokumentirati razlike, ki so vredne omembe,
    pogosto v več potrditvah, da jih sporočite
    na jasen način končnim uporabnikom.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignoriranje opustitve

  %p
    Ko ljudje nadgradijo z ene različice na drugo, bi moralo biti
    boleče jasno, kdaj se bo kaj zalomilo. Moralo bi biti mogoče
    nadgraditi na različico, ki navaja opustitve, odstraniti, kar je
    zastarelo, in nato nadgraditi na različico, kjer opustitve
    postanejo odstranitve.

  %p
    Če ne naredite nič drugega, v vašem dnevniku sprememb navedite opustitve,
    odstranitve in katere koli prelomne spremembe.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Zavajujoči datumi

  %p
    Območni formati datumov se po svetu razlikujejo in pogosto je
    težko najti človeku prijazen format datuma, ki se zdi intuitiven
    vsem. Prednost datumov, oblikovanih kot
    <code>2017-07-17</code> je, da sledijo vrstnemu redu od največje do
    najmanjše enote: leto, mesec in dan. Ta oblika se tudi ne
    prekriva na dvoumne načine z drugimi formati datumov, za razliko od nekaterih
    območnih formatov, ki zamenjajo položaj številk meseca in dneva.
    Ti razlogi in dejstvo, da je ta oblika datuma
    #{link_to "standardni ISO", data.links.isodate}, so to, kar naredijo to priporočeni
    format datuma za vnose v dnevnik sprememb.

  %aside
    Obstaja še več. Pomagajte nam zbrati te antivzorce, tako da
    = link_to "odprete težavo", data.links.issue
    ali zahtevek potega.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Pogosto zastavljena vprašanja

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Ali obstaja standardna oblika dnevnika sprememb?

  %p
    Niti ne. Obstaja #{link_to "stilski vodnik dnevnika sprememb GNU", data.links.gnustyle},
    ali pa "smernice" #{link_to "v dveh odstavkih dolgi datoteki GNU NEWS", data.links.gnunews}
    Oboje je neustrezno ali nezadostno.

  %p
    Ta projekt cilja biti
    = link_to "boljša konvencija dnevnika sprememb.", data.links.changelog
    Izhaja iz opazovanja dobrih praks v odprtokodni
    skupnosti in njihovega zbiranja.

  %p
    Zdrava kritika, razprava in predlogi za izboljšave
    = link_to "so dobrodošli.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Kako naj se imenuje datoteka dnevnika sprememb?

  %p
    Poimenujte jo <code>CHANGELOG.md</code>. Nekateri projekti uporabljajo
    <code>HISTORY</code>, <code>NEWS</code>, ali <code>RELEASES</code>.

  %p
    Čeprav je enostavno misliti, da ime vaše datoteke dnevnika sprememb
    ni tako pomembno, zakaj bi končnim uporabnikom otežili
    dosledno iskanje pomembnih sprememb?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Kaj pa GitHub Releases?

  %p
    To je odlična pobuda. #{link_to "Releases", data.links.github_releases} je mogoče uporabiti za 
    pretvorbo preprostih oznak Git (na primer oznaka z imenom <code>v1.0.0</code>)
    v bogate opombe ob izdaji z ročnim dodajanjem opomb izdaje ali pa lahko
    potegne sporočila anotiranih oznak Git in jih pretvori v opombe.

  %p
    GitHub Releases ustvarijo neprenosljiv dnevnik sprememb, ki ga je mogoče
    prikazati le uporabnikom v kontekstu GitHuba. Mogoče jih je
    narediti zelo podobne formatu Keep a Changelog, vendar
    je ponavadi potrebno nekoliko več vpletenosti.

  %p
    Trenutna različica izdaj GitHub tudi verjetno ni preveč
    odprta, da jo končni uporabniki odkrijejo, za razliko od tipičnih datotek z velikimi črkami
    (<code>README</code>, <code>CONTRIBUTING</code> itd.). Druga
    manjša težava je, da vmesnik trenutno ne ponuja povezav
    na dnevnike potrditev med vsako izdajo.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Ali je mogoče dnevnike sprememb samodejno razčleniti?

  %p
    Težko je, ker ljudje sledijo zelo različnim formatom in
    imenom datotek.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Rubyjev gem, ustvarjen pri
    ekipi Gemnasium in razčlenjuje mnoge (vendar
    ne vse) dnevnike sprememb odprtokodnih projektov.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Kaj pa izvlečene (angl. yanked) izdaje?

  %p
    Izvlečene izdaje so različice, ki jih je bilo treba umakniti zaradi
    resne napake ali varnostne težave. Pogosto se te različice sploh ne
    pojavijo v dnevnikih sprememb. Morale pa bi se. Prikazati bi jih morali
    tako:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Značka <code>[YANKED]</code> je v velikih črkah z razlogom. Pomembno
    je, da jo ljudje opazijo. Ker je obdana z oglatimi oklepaji, jo je tudi
    mogoče enostavneje programsko razčleniti.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Bi morali kdaj prepisati dnevnik sprememb?

  %p
    Seveda. Vedno obstajajo dobri razlogi za izboljšanje dnevnika sprememb.
    Redno odpiram zahteve potegov, da dodam manjkajoče izdaje odprtokodnim
    projektom z nevzdrževanimi dnevniki sprememb.

  %p
    Možno je tudi, da odkrijete, da ste pozabili obravnavati
    kritično spremembo v opombah za različico. Očitno je pomembno,
    da v tem primeru posodobite dnevnik sprememb.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Kako lahko prispevam?

  %p
    Ta dokument ni <strong>resnica</strong>; to je moje skrbno
    pretehtano mnenje, skupaj z informacijami in primeri, ki sem jih zbral.

  %p
    To je zato, ker želim, da naša skupnost doseže soglasje. Menim,
    da je razprava enako pomembna kot končni rezultat.

  %p
    Zato vas prosim, da <strong>#{link_to "se oglasite", data.links.repo}</strong>.

.press
  %h3 Pogovori
  %p
    Odšel sem na #{link_to "podcast The Changelog", data.links.thechangelog},
    da bi govoril o tem, zakaj bi morali vzdrževalci in sodelavci skrbeti za dnevnike sprememb,
    in tudi o motivacijah za tem projektom.
```

## File: source/sl/1.1.0/index.html.haml
```haml
---
title: Vzdržujte dnevnik sprememb
description: Ne dopustite, da vaši prijatelji odlagajo dnevnike Git v dnevnike sprememb.
language: sl
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Različica
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Kaj je dnevnik sprememb?

  %p
    Dnevnik sprememb (angl. changelog) je datoteka, ki vsebuje kuriran, kronološko
    urejen seznam opaznih sprememb za vsako različico projekta.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Zakaj vzdrževati dnevnik sprememb?

  %p
    Za poenostavitev, da uporabniki in sodelavci natančno vidijo, katere
    opazne spremembe so bile narejene med vsako izdajo (ali različico)
    projekta.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kdo potrebuje dnevnik sprememb?

  %p
    Ljudje to potrebujejo. Ne glede na to, ali so uporabniki ali razvijalci, so končni uporabniki
    programske opreme ljudje, ki jim je mar, kaj je v programski opremi. Ko se
    programska oprema spremeni, ljudje želijo vedeti, zakaj in kako.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Kako naredim dober dnevnik sprememb?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Smernice

  %ul
    %li
      Dnevniki sprememb so <em>za ljudi</em>, ne za stroje.
    %li
      Obstajati mora vnos za vsako posamezno različico.
    %li
      Enake vrste sprememb je treba združiti.
    %li
      Različice in razdelki morajo biti povezljivi.
    %li
      Najnovejša različica je na prvem mestu.
    %li
      Prikaže se datum izdaje vsake različice.
    %li
      Navedite, ali sledite #{link_to "Semantičnim različicam", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Vrste sprememb

  %ul
    %li
      %code Dodano (angl. Added)
      za nove lastnosti.
    %li
      %code Spremenjeno (angl. Changed)
      za spremembe obstoječe funkcionalnosti.
    %li
      %code Opuščeno (angl. Deprecated)
      za lastnosti, ki bodo kmalu odstranjene.
    %li
      %code Odstranjeno (angl. Removed)
      za zdaj odstranjene lastnosti.
    %li
      %code Popravljeno (angl. Fixed)
      za morebitne popravke napak.
    %li
      %code Varnost (angl. Security)
      v primeru ranljivosti.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Kako lahko zmanjšam trud, potreben za vzdrževanje dnevnika sprememb?

  %p
    Na vrhu imejte razdelek <code>Neobjavljeno</code>, če želite slediti prihajajočim
    spremembam.

  %p To ima dva namena:

  %ul
    %li
      Ljudje lahko vidijo, kakšne spremembe lahko pričakujejo v prihajajočih izdajah
    %li
      Ob izdaji lahko premaknete razdelek sprememb <code>Neobjavljeno</code>
      v nov razdelek različice izdaje.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Ali so lahko dnevniki sprememb slabi?

  %p Da. Tukaj je nekaj načinov, kako so lahko manj koristni.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Razlike dnevnika potrditev (angl. diffs)

  %p
    Uporaba razlik dnevnika potrditev kot dnevnikov sprememb je slaba ideja: polni so
    navlake. Stvari, kot so potrditve združitev, potrditve z nejasnimi naslovi,
    spremembe dokumentacije itd.

  %p
    Namen potrditve je dokumentirati korak v razvoju
    izvorne kode. Nekateri projekti imajo potrditve čiste, nekateri ne.

  %p
    Namen vnosa v dnevnik sprememb je dokumentirati razlike, ki so vredne omembe,
    pogosto v več potrditvah, da jih sporočite
    na jasen način končnim uporabnikom.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignoriranje opustitve

  %p
    Ko ljudje nadgradijo z ene različice na drugo, bi moralo biti
    boleče jasno, kdaj se bo kaj zalomilo. Moralo bi biti mogoče
    nadgraditi na različico, ki navaja opustitve, odstraniti, kar je
    zastarelo, in nato nadgraditi na različico, kjer opustitve
    postanejo odstranitve.

  %p
    Če ne naredite nič drugega, v vašem dnevniku sprememb navedite opustitve,
    odstranitve in katere koli prelomne spremembe.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Zavajujoči datumi

  %p
    Območni formati datumov se po svetu razlikujejo in pogosto je
    težko najti človeku prijazen format datuma, ki se zdi intuitiven
    vsem. Prednost datumov, oblikovanih kot
    <code>2017-07-17</code> je, da sledijo vrstnemu redu od največje do
    najmanjše enote: leto, mesec in dan. Ta oblika se tudi ne
    prekriva na dvoumne načine z drugimi formati datumov, za razliko od nekaterih
    območnih formatov, ki zamenjajo položaj številk meseca in dneva.
    Ti razlogi in dejstvo, da je ta oblika datuma
    #{link_to "standardni ISO", data.links.isodate}, so to, kar naredijo to priporočeni
    format datuma za vnose v dnevnik sprememb.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Nedosledne spremembe

  %p
    Dnevnik sprememb, ki omenja samo nekatere spremembe, je lahko enako nevaren,
    kot če nimate dnevnika sprememb. Čeprav mnoge spremembe morda niso
    pomembne - na primer, odstranitve enega samega presledka morda ne bo treba
    zabeležiti v vseh primerih - je pa treba vse pomembne spremembe
    omeniti v dnevniku sprememb. Z nedoslednim uveljavljanjem sprememb
    lahko vaši uporabniki zmotno mislijo, da je dnevnik sprememb edini vir
    resnice. Moral bi biti. Z veliko močjo prihaja tudi velika odgovornost -
    imeti dober dnevnik sprememb pomeni imeti dosledno posodobljen dnevnik sprememb.

  %aside
    Obstaja še več. Pomagajte nam zbrati te antivzorce, tako da
    = link_to "odprete težavo", data.links.issue
    ali zahtevek potega.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Pogosto zastavljena vprašanja

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Ali obstaja standardna oblika dnevnika sprememb?

  %p
    Niti ne. Obstaja #{link_to "stilski vodnik dnevnika sprememb GNU", data.links.gnustyle},
    ali pa "smernice" #{link_to "v dveh odstavkih dolgi datoteki GNU NEWS", data.links.gnunews}
    Oboje je neustrezno ali nezadostno.

  %p
    Ta projekt cilja biti
    = link_to "boljša konvencija dnevnika sprememb.", data.links.changelog
    Izhaja iz opazovanja dobrih praks v odprtokodni
    skupnosti in njihovega zbiranja.

  %p
    Zdrava kritika, razprava in predlogi za izboljšave
    = link_to "so dobrodošli.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Kako naj se imenuje datoteka dnevnika sprememb?

  %p
    Poimenujte jo <code>CHANGELOG.md</code>. Nekateri projekti uporabljajo
    <code>HISTORY</code>, <code>NEWS</code>, ali <code>RELEASES</code>.

  %p
    Čeprav je enostavno misliti, da ime vaše datoteke dnevnika sprememb
    ni tako pomembno, zakaj bi končnim uporabnikom otežili
    dosledno iskanje pomembnih sprememb?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Kaj pa GitHub Releases?

  %p
    To je odlična pobuda. #{link_to "Releases", data.links.github_releases} je mogoče uporabiti za 
    pretvorbo preprostih oznak Git (na primer oznaka z imenom <code>v1.0.0</code>)
    v bogate opombe ob izdaji z ročnim dodajanjem opomb izdaje ali pa lahko
    potegne sporočila anotiranih oznak Git in jih pretvori v opombe.

  %p
    GitHub Releases ustvarijo neprenosljiv dnevnik sprememb, ki ga je mogoče
    prikazati le uporabnikom v kontekstu GitHuba. Mogoče jih je
    narediti zelo podobne formatu Keep a Changelog, vendar
    je ponavadi potrebno nekoliko več vpletenosti.

  %p
    Trenutna različica izdaj GitHub tudi verjetno ni preveč
    odprta, da jo končni uporabniki odkrijejo, za razliko od tipičnih datotek z velikimi črkami
    (<code>README</code>, <code>CONTRIBUTING</code> itd.). Druga
    manjša težava je, da vmesnik trenutno ne ponuja povezav
    na dnevnike potrditev med vsako izdajo.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Ali je mogoče dnevnike sprememb samodejno razčleniti?

  %p
    Težko je, ker ljudje sledijo zelo različnim formatom in
    imenom datotek.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Rubyjev gem, ustvarjen pri
    ekipi Gemnasium in razčlenjuje mnoge (vendar
    ne vse) dnevnike sprememb odprtokodnih projektov.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Kaj pa izvlečene (angl. yanked) izdaje?

  %p
    Izvlečene izdaje so različice, ki jih je bilo treba umakniti zaradi
    resne napake ali varnostne težave. Pogosto se te različice sploh ne
    pojavijo v dnevnikih sprememb. Morale pa bi se. Prikazati bi jih morali
    tako:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Značka <code>[YANKED]</code> je v velikih črkah z razlogom. Pomembno
    je, da jo ljudje opazijo. Ker je obdana z oglatimi oklepaji, jo je tudi
    mogoče enostavneje programsko razčleniti.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Bi morali kdaj prepisati dnevnik sprememb?

  %p
    Seveda. Vedno obstajajo dobri razlogi za izboljšanje dnevnika sprememb.
    Redno odpiram zahteve potegov, da dodam manjkajoče izdaje odprtokodnim
    projektom z nevzdrževanimi dnevniki sprememb.

  %p
    Možno je tudi, da odkrijete, da ste pozabili obravnavati
    kritično spremembo v opombah za različico. Očitno je pomembno,
    da v tem primeru posodobite dnevnik sprememb.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Kako lahko prispevam?

  %p
    Ta dokument ni <strong>resnica</strong>; to je moje skrbno
    pretehtano mnenje, skupaj z informacijami in primeri, ki sem jih zbral.

  %p
    To je zato, ker želim, da naša skupnost doseže soglasje. Menim,
    da je razprava enako pomembna kot končni rezultat.

  %p
    Zato vas prosim, da <strong>#{link_to "se oglasite", data.links.repo}</strong>.

.press
  %h3 Pogovori
  %p
    Odšel sem na #{link_to "podcast The Changelog", data.links.thechangelog},
    da bi govoril o tem, zakaj bi morali vzdrževalci in sodelavci skrbeti za dnevnike sprememb,
    in tudi o motivacijah za tem projektom.
```

## File: source/sr/1.0.0/index.html.haml
```haml
---
title: Vodite changelog
description: Ne dajte prijateljima da trpaju git logove u changelogove.
language: sr
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Verzija
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Šta je changelog?

  %p
    Changelog je datoteka koja sadrži uređeni hronološki
    poređani popis značajnih promena unutar svake verzije projekta.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Zašto voditi changelog?

  %p
    Kako bi korisnici i saradnici detaljno videli
    značajne promjene među pojedinim izdanjima (ili verzijama)
    projekta.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kome treba changelog?

  %p
    Ljudima. Bilo da su uobičajeni korisnici ili programeri, krajnji su korisnici
    softvera ljudska bića kojima je stalo do toga od čega je sastavljen. Kada
    se softver menja, korisnici žele znati kako i zašto.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Kako kreirati dobar changelog?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Glavna načela

  %ul
    %li
      Changelogs služi <em> ljudima</em>, ne mašinama.
    %li
      Potrebno je stvoriti unos za svaku verziju.
    %li
      Slične promene potrebno je grupisati.
    %li
      Verzije i odeljci trebaju imati vezu.
    %li
      Poslednja verzija treba biti na prvom mestu.
    %li
      Datum izdavanja svake pojedine verzije treba biti vidljiv.
    %li
      Navesti prati li se #{link_to "Semantičko verzioniranje", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Vrste promena

  %ul
    %li
      %code Added
      za nove funkcionalnosti.
    %li
      %code Changed
      za promene u postojećim funkcionalnostima.
    %li
      %code Deprecated
      za funkcionalnosti koje će se ukloniti u budućim verzijama.
    %li
      %code Removed
      za uklonjene funkcionalnosti.
    %li
      %code Fixed
      za ispravke bagova.
    %li
      %code Security
      za slučaj sigurnosnih propusta.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Kako održavati changelog sa što manje napora?

  %p
    Na vrh postavite <code>Unreleased</code> sekciju gde ćete navoditi nadolazeće
    promene.

  %p To radimo iz dva razloga:

  %ul
    %li
      Korisnici mogu videti promene koje mogu očekivati u nadolazećim izdanjima
    %li
      Kod izdavanja nove verzije, moguće je <code>Unreleased</code> sekciju
      samo preimenovati u novo izdanje.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Može li changelog biti loš?

  %p Naravno. Postoji nekoliko slučajeva u kojima changelog može biti beskoristan.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit log diffovi

  %p
    Korištenje commit log diffova u svrhu changeloga nije dobra ideja:
    puni su šuma. Npr. merge commitovi, commitovi s nejasnim naslovima,
    promene u dokumentaciji i sl.

  %p
    Svrha commita je da beleži korake u razvoju izvornog koda.
    U nekim projektima se comittovi čiste, no ponekad i ne.

  %p
    Svrha unosa u changelogu je da beleži značajne razlike, a
    često kroz više commitova i jasno ih prenese krajnjem
    korisniku.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorisanje uklonjenih funkcionalnosti

  %p
    Kad korisnici nadograde softver na noviju verziju, treba biti potpuno
    jasno da postoji mogućnost da će se neki deo pokvariti. Softver treba biti
    moguće nadograditi na verziju koja će navesti funkcionalnosti koje trebaju
    biti uklonjene, uklanja takve te se kasnije može nadograditi na verziju
    gde su već uklonjene.

  %p
    U najmanju ruku, potrebno je u changelogu navoditi funkcionalnosti koje će
    biti uklonjene, funkcionalnosti koje su uklonjene i promene koje će
    uticati na rad softvera (breaking change).


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Nejasni datumi

  %p
    Regionalni formati datuma variraju delom sveta, pa je često
    teško pronaći format datuma koji će odgovarati svima.
    Prednost datuma u formatu <code>2017-07-17</code> je to što je poređan
    od veće prema manjoj jedinici: godina, mesec, dan. Ovaj je format takođe
    teško zameniti s drugim regionalnim formatima, za razliku od nekih koji,
    primera radi, menjaju poziciju oznake meseca i dana.
    Iz tog razloga, a i zbog toga što je navedeni format #{link_to "ISO standard", data.links.isodate},
    taj se format preporučuje za changelog unose.

  %aside
    Ali to nije sve. Pomozite u prikupljanju primera loše prakse
    = link_to "otvorivši issue", data.links.issue
    ili pull request.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Česta pitanja

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Postoji li standardni changelog format?

  %p
    Zapravo ne. Postoji #{link_to "GNU changelog stilski priručnik", data.links.gnustyle},
    i #{link_to "GNU NEWS file", data.links.gnunews}
    "priručnik od dva odlomka". Nijedan nije adekvatan ni dovoljan.

  %p
    Cilj ovoga projekta
    = link_to "kvalitetniji changelog standard.", data.links.changelog
    Nastao je prikupljanjem primera dobra prakse u
    open source zajednici.

  %p
    Konstruktivna kritika, raprave i predlozi za poboljšanje
    = link_to "su dobrodošli.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Kako nazvati changelog datoteku?

  %p
    Dajemo joj naziv <code>CHANGELOG.md</code>. Neki projekti još koriste
    <code>HISTORY</code>, <code>NEWS</code> ili <code>RELEASES</code>.

  %p
    Iako može delovati da naziv changelog datoteke i nije toliko
    bitan, čemu otežavati korisnicima da dođu do informacije o promjenama?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Šta s GitHub Releases?

  %p
    To je ozbiljna inicijativa. #{link_to "Releases", data.links.github_releases} se mogu koristiti
    kako bi git oznake (npr. git oznaka <code>v1.0.0</code>) pretvorili
    u opširnije beleške o izdanju, upisujući ih ručno ili pak preuzimajući
    anotirane git oznake i pretvarajući ih u unose.

  %p
    GitHub Releases stvara statični changelog vidljiv korisnicima
    unutar GitHub repozitorijuma. Moguće ih je urediti u format koji
    bi odgovarao Vodite changelog formatu, no često je nešto opširniji.

  %p
    Trenutna GitHub releases verzija i nije baš sasvim vidljiva
    korisnicima, za razliku od uobičajenih datoteka označenih velikim slovima
    (<code>README</code>, <code>CONTRIBUTING</code>, itd.). Još je jedan
    manji problem što trenutno interfejs ne nudi poveznice na commit logove
    između izdanja.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Mogu li changelogovi automatski parsirati?

  %p
    Teže, jer se koriste vrlo različiti formati, kao i nazivi datoteka.

  %p
    #{link_to "Vandamme", data.links.vandamme} je Ruby gem koji je kreirao
    Gemnasium tim i može parsirati mnoge (ali
    ne sve) changelogove projekata otvorenog koda.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Šta s povučenim izdanjima?

  %p
    Povučena ili 'Yanked' izdanja su verzije koje su uklonjene zbog
    ozbiljnijeg baga ili sigurnosnog propusta. Takva se izdanja najčešće
    i ne pojavljuju u changelogu, iako bi trebala. Trebala bi biti navedena
    na sledeći način:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> oznaka jasno je istaknuta s razlogom. Bitno je
    da ju je lako primetiti. Buduće da je okružena zagradama, takođe ju
    je lakše parsirati.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Da li je potrebno prepravljati changelog?

  %p
    Naravno. Često postoje dobri razlozi da bismo poboljšali changelog. Ja
    često otvaram pull requestove kako bih dodao nedostajuća izdanja projektima
    otvorenog koda, koji ne održavaju changelogove.

  %p
    Moguće je, takođe da otkrijete, kako ste zaboravili navesti promenu koja
    bi uticala na rad (breaking change). U tom slučaju je, očito, vrlo bitno
    ažurirati changelog.


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Kako doprineti?

  %p
    Ovaj dokument nije <em>Sveto Pismo</em>; ovo je samo pažljivo
    razmotreno mišljenje, uz informacije i primere koje sam skupio.

  %p
    Razlog je tome to što želim da zajednica postigne konsenzus. Verujem,
    takođe, da je i sama rasprava bitna kao i krajnji rezultat.

  %p
    Zato, molimo <strong>#{link_to "uskočite", data.links.repo}</strong>.

.press
  %h3 Razgovori
  %p
    Gostovao sam na #{link_to "The Changelog podcastu", data.links.thechangelog}
    gde sam pokušao objasniti zašto začetnici projekata i saradnici trebaju
    brinuti o changelogovima te motivaciji iza ovog projekta.
```

## File: source/sv/0.3.0/index.html.haml
```haml
---
title: Håll en ändringslogg
description: Låt inte dina vänner slänga in en git logs i CHANGELOG™
language: sv
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Vad är en ändringslogg?
  En ändringslogg är en fil innehållandes en sammanfattad, kronologiskt ordnad
  lista över ändringar för varje version av ett projekt.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Vad är meningen med en ändringslogg?
  För att göra det enkelt för användare och medarbetare att se exakt vilka
  viktiga ändringar som har gjorts mellan varje utgåva (eller version) av projektet.

  ### Varför ska jag bry mig?
  Därför att mjukvaruverktyg är för människor. Om du inte bryr dig, varför
  bidrar du till öppen källkod? Visst finns det väl någon del i din vackra
  hjärna som bryr sig.

  Jag [pratade med Adam Stacoviak och Jerod Santo från podcasten The Changelog][thechangelog]
  (passande, eller hur?) om varför ansvariga och bidragsgivare bör bry sig,
  och motiveringen bakom detta projekt.
  Om du kan avvara lite tid (1:06), rekommenderar jag att lyssna på det.

  ### Vad gör en bra ändringslogg?
  Jag är glad att du frågade.

  En bra ändringslogg håller sig till dessa principer:

  - Den är skapad för människor, inte maskiner, så läsbarhet är avgörande.
  - Lätt att länka till valfri sektion (därav Markdown framför klartext).
  - En undersektion per version.
  - Listar utgåvor i omvänd kronologisk ordning (nyast högst upp).
  - Anger alla datum på formatet `YYYY-MM-DD`
  (exempel: `2012-06-02` för 2:a juni 2012). Det är internationellt,
  [förnuftigt](https://xkcd.com/1179/) och språkoberoende.
  - Anger uttryckligen om projektet följer [Semantisk versionshantering][SemVer].
  - Varje version bör:
    - Ange datum då utgåvan släpptes på formatet angivet ovan.
    - Gruppera ändringar för att beskriva deras inverkan på projektet enligt följande:
      - `Added` för nya funktioner.
      - `Changed` för ändringar på existerande funktionalitet.
      - `Deprecated` för tidigare stabila funktioner som tas bort i nästa utgåva.
      - `Removed` för funktioner tidigare markerade som `Deprecated` som tas bort i denna version.
      - `Fixed` för buggfixar.
      - `Security` för att uppmana användare att uppgradera vid fall av sårbarheter.

  ### Hur kan jag minimera den insats som krävs?
  Ha alltid en sektion kallad `"Unreleased"` högst upp för att hålla reda på
  alla ändringar.

  Detta tjänar två syften:

  - Folk kan se vilka ändringar som kan förväntas i kommande utgåvor
  - Vid lansering behöver du bara ändra `"Unreleased"` till versionsnumret
    och lägga till en ny `"Unreleased"` högst upp.

  ### Vad får änglarna att gråta?
  Okej...låt oss gå genom det.

  - **Dumpa ut en diff från commit-loggen.** Gör helt enkelt inte så, du hjälper ingen.
  - **Inte betona `deprecations`.** När användare uppgraderar från en version till
    en annan ska det vara smärtsamt uppenbart om något förväntas gå sönder.
  - **Datum i region-specifikt format.** I USA lägger folk månaden först
    ("06-02-2012" för 2:a juni 2012, vilket bara är *konstigt*), medan många
    andra runt om i världen skriver "2 June 2012" men uttalar det annorlunda.
    "2012-06-02" fungerar logiskt från största till minsta, inte överlappar på ett
    tvetydligt sätt med andra datumformat, och det är en
    [ISO-standard](http://www.iso.org/iso/home/standards/iso8601.htm). Dessutom
    är det rekommenderat datumformat för ändringsloggar.


  Det finns mer. Hjälp mig att samla tårarna från änglarna genom att
  [öppna en issue][issues]
  eller en pull-förfrågan.

  ### Finns det ett standardformat på ändringsloggar?
  Tyvärr inte. Men lugn. Jag vet att du frustrerad kommer att rusa iväg för att hitta
  den där länken till GNU:s format för ändringsloggar, eller den två stycken långa
  GNU NEWS-filen med "guideline". Stilguiden från GNU är en bra start, men den är
  tyvärr allt för naiv. Det är inget fel med att vara naiv, men när människor
  behöver vägledning är det inte speciellt hjälpsamt. Speciellt när det är många
  olika situationer och specialfall att hantera.

  Detta projekt [innehåller vad jag hoppas kommer att bli en bättre filkonvention
  för CHANGELOG][CHANGELOG]. Jag tror inte att status quo är tillräckligt bra,
  och jag tror att vi tillsammans kan komma fram till en bättre konvention
  om vi försöker att plocka ut bra erfarenheter från riktiga mjukvaruprojekt.
  Titta runt och kom ihåg att [diskussioner och förbättringsförslag är välkomna][issues]!

  ### Vad bör filen med ändringsloggen heta?
  Tja, om du inte kan lista ut det från exemplen ovan är `CHANGELOG.md`
  den bästa konvention hittills.

  En del projekt använder också `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`, etc.

  Det är en verklig röra. Alla dessa namn gör det bara svårare för människor att hitta.

  ### Varför kan folk inte bara använda en diff från `git log`?
  Eftersom logg-diffar av naturen är fulla med brus. Det kan inte ens användas
  för att göra en lämplig ändringslogg ens i ett hypotetiskt projekt drivet av
  perfekta människor som aldrig skriver fel, aldrig glömmer att checka in nya filer
  eller aldrig glömmer någon del av en refaktorering. Syftet med en commit är att
  dokumentera ett enskilt steg i processen då koden utvecklas från ett tillstånd till
  ett annat. Syftet med en ändringslogg är att dokumentera de anmärkningsvärda
  skillnaderna mellan dessa tillstånd.

  På samma sätt som det är skillnad mellan bra kommentarer och själva koden,
  är det skillnad mellan ändringsloggen och commit-loggen:
  en beskriver *varför* och den andra *hur*.

  ### Kan ändringsloggar bli automatiskt tolkade?
  Det är svårt då människor följer vitt olika format och filnamn.

  [Vandamme][vandamme] är en Ruby gem
  skapad av gruppen bakom [Gemnasium][gemnasium] som tolkar många (men inte alla)
  ändringsloggar för öppen källkod.

  ### Varför alternerar du mellan att skriva det som "CHANGELOG" och "ändringslogg"?
  "CHANGELOG" är namnet på själva filen. Det sticker ut lite, men det är en
  historisk konvention använt i många öppna källkodsprojekt. Andra
  exempel på liknande filer är [`README`][README], [`LICENSE`][LICENSE]
  och [`CONTRIBUTING`][CONTRIBUTING].

  De stora bokstäverna i namnen (som gjorde att de i äldre operativsystem
  hamnade högst upp) används för att dra uppmärksamhet till dem. Då de är
  viktig metadata om projektet borde de vara användbara för att alla som
  vill använda eller bidra till det, ungefär som [open source project badges][shields].

  När jag refererar till "ändringslogg" pratar jag om syftet med denna fil:
  att logga ändringar.

  ### Hur är det med brådskande utgåvor ("yanked")?
  Brådskande utgåvor är versioner som måste släppas p.g.a. alvarliga
  buggar eller säkerhetsproblem. Oftast brukar dessa versioner inte ens
  finnas med i ändringsloggarna. De borde det. Så här borde du visa dem:

  `## [0.0.5] - 2014-12-13 [YANKED]`

  Taggen `[YANKED]` står ut av en anledning, det är viktigt för människor
  att se den. Då den är omsluten med hakparenteser är det också lättare
  för ett program att tolka.

  ### Borde du någonsin förändra en ändringslogg?
  Självklart. Det finns alltid en bra anlending att förbättra en ändringslogg.
  Jag öppnar regelbundet pull requests för att lägga till saknade utgåvor
  för öppna källkodsprojekt som inte tar hand om sin ändringslogg.

  Det kan också hända att du upptäcker att du glömt att ta upp en icke
  bakåtkompatibel förändring i en version. I sådana fall är det självklart
  viktigt att uppdatera din ändringslogg.

  ### Hur kan jag bidra?
  Detta dokument är inte **sanningen**, det är en noga övervägd åsikt
  tillsammans med information och exempel jag har samlat på mig.
  Även om jag tillhandahåller en [CHANGELOG][] i min [GitHub repo][gh],
  har jag avsiktligt inte skapat en egentlig *utgåva* eller en tydlig lista
  med regler att följa (som t.ex. [SemVer.org][semver] gör).

  Detta beror på att jag vill att vår gemenskap ska nå samförstånd. Jag
  tror att diskussionen är lika viktig som slutresultatet.

  Så välkomna att [**diskutera**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/sv/1.0.0/index.html.haml
```haml
---
title: För en ändringslogg
description: Låt inte dina vänner slänga in git-loggar i ändringsloggar.
language: sv
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Vad är en ändringslogg?

  %p
    En ändringslogg är en fil innehållandes en sammanfattad, kronologiskt ordnad
    lista över viktiga ändringar för varje version av ett projekt.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Vad är meningen med en ändringslogg?

  %p
    För att göra det enklare för användare och medarbetare att se exakt vilka
    viktiga ändringar som har gjorts mellan varje utgåva (eller version) av projektet.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Vem behöver en ändringslogg?

  %p
    Alla behöver det. Oavsett om det är användare eller utvecklare, är
    alla slutanvändare av mjukvaran människor som bryr sig vad som finns
    i mjukvaran. När mjukvaran förändras vill folk veta varför och hur.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Hur gör jag en bra ändringslogg?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Riktlinjer

  %ul
    %li
      Ändringsloggar är <em>för människor</em>, inte maskiner.
    %li
      Det bör finnas en post för varje enskild version.
    %li
      Samma typ av ändringar bör grupperas.
    %li
      Det bör vara möjligt att länka till versioner och sektioner.
    %li
      Senaste versionen kommer först.
    %li
      Datum då respektive version släpptes ska visas.
    %li
      Notering att du följer #{link_to "Semantisk versionshantering", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Typer av ändringar

  %ul
    %li
      %code Added
      för nya funktioner.
    %li
      %code Changed
      för ändringar på existerande funktionalitet.
    %li
      %code Deprecated
      för funktionalitet som snart tas bort.
    %li
      %code Removed
      för nu borttagen funktionalitet.
    %li
      %code Fixed
      för alla buggfixar
    %li
      %code Security
      i fall av sårbarheter.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Hur kan jag minimera den insats som krävs för att underhålla en ändringslogg?

  %p
    Ha en sektion kallad <code>Unreleased</code> högst upp för att hålla reda på
    kommande ändringar.

  %p Detta tjänar två syften:

  %ul
    %li
      Folk kan se vilka ändringar de kan förvänta sig i kommande utgåvor
    %li
      Vid lansering behöver du bara flytta innehållet i sektionen
      <code>Unreleased</code> till en ny versionspost.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kan ändringsloggar vara dåliga?

  %p Ja, här är några exempel på då de är mindre användbara.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffar från incheckningsloggen.

  %p
    Det är en dålig idé att använda incheckningsloggen som ändringslogg:
    de är fulla av brus. Saker så som merge-incheckningar, incheckningar med
    otydliga titlar, dokumentationsförändringar etc.

  %p
    Syftet med en incheckning är att dokumentera ett steg i utvecklingen av
    källkoden. Vissa projekt städar upp bland incheckningarna, andra inte.

  %p
    Syftet med en post i en ändringslogg är att dokumentera den noterbara
    skillnaden, oftast över flera incheckningar, för att kommunicera dessa
    tydligt till slutanvändaren.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorera föråldrad funktionalitet (deprecations)

  %p
    När användare uppgraderar från en version till en annan, ska det vara
    smärtsamt uppenbart när något förväntas gå sönder. Den bör vara möjligt
    att uppgradera till en version som listar föråldrad funktionalitet, ta
    bort dessa beroenden i sitt program, och sedan uppgradera till en version
    där den föråldrade funktionaliteten är borttagen.

  %p
    Om du inte gör något annat, lista åtminstone föråldrad och borttagen
    funktionalitet samt förstörande förändringar i din ändringslogg.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Förvillande datum

  %p
    Lokala datumformat varierar över hela världen, och det är ofta
    svårt att hitta ett användbart datumformat som känns intuitivt
    för alla. Fördelen med datumformat så som
    <code>2017-07-17</code> är att det följer storleksordningen från störst till
    minst: år, månad och dag. Detta format överlappar inte heller
    på ett tvetydligt sätt med andra datumformat, till skillnad från
    lokala format som kan växla positionen på månad och dag.
    Dessa anledningar tillsammans med det faktum att detta datumformat är en
    #{link_to "ISO-standard", data.links.isodate}, gör att detta är rekommenderat
    format för ändringsloggar.

  %aside
    Det finns mer. Hjälp mig att samla dessa antimönster genom att
    = link_to "skapa ett issue", data.links.issue
    eller en pull requests

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Vanliga frågor

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Finns det ett standardformat på ändringsloggar?

  %p
    Inte riktigt. #{link_to "GNU:s stilguide för ändringsloggar", data.links.gnustyle} och
    den #{link_to "två stycke långa GNU NEWS-filen", data.links.gnunews} med "riktlinjer" finns.
    Båda är bristfälliga och otillräckliga.

  %p
    Detta projekt har som mål att bli
    = link_to "en bättre konvention för ändringsloggar.", data.links.changelog
    Det utgår från uppenbart goda praxis från öppen källkods-världen och sammanför dem.

  %p
    Konstruktiv kritik, diskussion och förslag till förbättring
    = link_to "är välkommen.", data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Vad bör filen med ändringsloggen heta?

  %p
    Döp den till <code>CHANGELOG.md</code>. En del projekt använder
    <code>HISTORY</code>, <code>NEWS</code> eller <code>RELEASES</code>.

  %p
    Även om det är lätt att tänka att det inte spelar så stor roll vad filen
    med ändringsloggar kallas, varför göra det svårare för dina slutanvändare
    att enkelt hitta de viktigaste ändringarna?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Hur är det med GitHub Releases?

  %p
    Det är ett fantasiskt initiativ. #{link_to "Releases", data.links.github_releases} kan användas
    för att göra enkla git-taggar (t.ex. en tagg kallad <code>v1.0.0</code>)
    till formaterade versionsanteckningar genom att manuellt lägga till
    versionsanteckningar eller så kan den hämta meddelandena i kommenterade
    git-taggar och omvandla dessa till versionsanteckningar.

  %p
    GitHub Releases skapar en icke porterbar ändringslogg som enbart kan visas
    för användare på GitHub. Det är möjligt att formatera det ungefär som på
    För en ändringslogg-formatet, men det tendera att bli lite mer invecklat.

  %p
    Nuvarande version av GitHub releases är möjligtvis också lite svår att
    hitta för slutanvändare, till skillnad från filer med normalt stora
    bokstäver (<code>README</code>, <code>CONTRIBUTING</code>, etc.).
    Ett annat bekymmer är att användargränssnittet för närvarande inte
    erbjuder länkar till incheckningsloggar mellan olika versioner.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Kan ändringsloggar bli automatiskt tolkade?

  %p
    Det är svårt då människor följer vitt olika format och filnamn.

  %p
    #{link_to "Vandamme", data.links.vandamme} är en Ruby gem skapad av gruppen
    Gemnasium som tolkar många (men inte alla)
    ändringsloggar för öppen källkod.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Hur är det med återtagna utgåvor ("yanked")?

  %p
    Återtagna utgåvor är versioner som måste tas tillbaka på grund av
    allvarliga buggar eller säkerhetsproblem. Oftast brukar dessa versioner
    inte ens finnas med i ändringsloggarna. De borde det. Så här borde du
    visa dem:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Taggen <code>[YANKED]</code> står ut av en anledning, det är viktigt
    att folk se det. Då den är omsluten med hakparenteser är det också lättare
    för ett program att tolka det.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Borde du någonsin förändra en ändringslogg?

  %p
    Självklart. Det finns alltid en bra anledning att förbättra en ändringslogg.
    Jag öppnar regelbundet pull requests för att lägga till saknade utgåvor
    för öppna källkodsprojekt som inte tar hand om sin ändringslogg.

  %p
    Det kan också hända att du upptäcker att du glömt att ta upp en icke
    bakåtkompatibel förändring i en version. I sådana fall är det självklart
    viktigt att uppdatera din ändringslogg.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Hur kan jag bidra?

  %p
    Detta dokument är inte <strong>sanningen</strong>, det är en noga övervägd
    åsikt tillsammans med information och exempel jag har samlat på mig.

  %p
    Detta beror på att jag vill att vår gemenskap ska nå enighet. Jag tror på
    att diskussionen är lika viktig som slutresultatet.

  %p
    Så tveka inte att <strong>#{link_to "kasta dig in i diskussionen", data.links.repo}</strong>.

.press
  %h3 Samtal
  %p
    Jag var med i #{link_to "The Changelog podcast", data.links.thechangelog}
    för att prata om varför förvaltare och bidragsgivare bör bry sig om
    ändringsloggar, och motiveringen bakom detta projekt.
```

## File: source/sv/1.1.0/index.html.haml
```haml
---
title: För en ändringslogg
description: Låt inte vänner dumpa git-loggar i ändringsloggar.
language: sv
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Vad är en ändringslogg?

  %p
    En ändringslogg är en fil innehållandes en sammanfattad, kronologiskt ordnad
    lista över viktiga ändringar för varje version av ett projekt.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Varför föra en ändringslogg?

  %p
    För att göra det enklare för användare och medarbetare att se exakt vilka
    viktiga ändringar som har skett mellan varje utgåva (eller version) av projektet.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Vem behöver en ändringslogg?

  %p
    Folk i allmänhet. Oavsett om de är användare eller utvecklare, är
    alla slutanvändare av mjukvaran människor som bryr sig om vad som finns
    i den. När mjukvaran förändras vill de veta varför och hur.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Hur skapar jag en bra ändringslogg?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Riktlinjer

  %ul
    %li
      Ändringsloggar är till <em>för människor</em>, inte maskiner.
    %li
      Det bör finnas en post för varje enskild version.
    %li
      Samma typ av ändringar bör grupperas.
    %li
      Det bör vara möjligt att länka till versioner och sektioner.
    %li
      Senaste versionen kommer först.
    %li
      Datum då respektive version släpptes ska visas.
    %li
      Nämn huruvida du använder #{link_to "Semantisk versionshantering", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Ändringstyper

  %ul
    %li
      %code Added
      för nya funktioner.
    %li
      %code Changed
      för ändringar i existerande funktionalitet.
    %li
      %code Deprecated
      för funktionalitet som snart tas bort.
    %li
      %code Removed
      för borttagen funktionalitet.
    %li
      %code Fixed
      för felrättningar
    %li
      %code Security
      i fall av sårbarheter.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Hur kan jag minimera den insats som krävs för att underhålla en ändringslogg?

  %p
    Ha en sektion kallad <code>Unreleased</code> högst upp för att hålla reda på
    kommande ändringar.

  %p Detta tjänar två syften:

  %ul
    %li
      Folk kan se vilka ändringar de kan förvänta sig i kommande utgåvor
    %li
      Vid lansering behöver du bara flytta innehållet i sektionen
      <code>Unreleased</code> till en ny versionspost.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Kan ändringsloggar vara dåliga?

  %p Ja, här är några exempel på då de är mindre användbara.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Diffar från incheckningsloggen.

  %p
    Det är en dålig idé att använda incheckningsloggen som ändringslogg:
    de är fulla av brus; merge-incheckningar, incheckningar med
    otydliga titlar, dokumentationsförändringar etc.

  %p
    Syftet med en incheckning är att dokumentera ett steg i utvecklingen av
    källkoden. Vissa projekt städar upp bland incheckningarna, andra inte.

  %p
    Syftet med en post i en ändringslogg är att dokumentera den märkbara
    skillnaden, oftast över flera incheckningar, för att kommunicera dessa
    tydligt till slutanvändaren.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ignorera föråldrad funktionalitet

  %p
    När användare uppgraderar från en version till en annan, ska det vara
    smärtsamt uppenbart när något förväntas gå sönder. Det bör vara möjligt
    att uppgradera till en version som listar föråldrad funktionalitet, ta
    bort dessa beroenden i sitt program, och sedan uppgradera till en version
    där den föråldrade funktionaliteten är borttagen.

  %p
    Även om du inte gör något annat, så lista åtminstone föråldrad och borttagen
    funktionalitet samt förstörande förändringar i din ändringslogg.

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Förvillande datum

  %p
    Lokala datumformat varierar över hela världen, och det är ofta
    svårt att hitta ett användbart datumformat som känns intuitivt
    för alla. Fördelen med datumformat så som
    <code>2017-07-17</code> är att det följer storleksordningen från störst till
    minst: år, månad och dag. Detta format överlappar inte heller
    på ett tvetydligt sätt med andra datumformat, till skillnad från
    lokala format som kan växla positionen på månad och dag.
    Dessa anledningar tillsammans med det faktum att detta datumformat är en
    #{link_to "ISO-standard", data.links.isodate}, gör att detta är rekommenderat
    format för ändringsloggar.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Inkonsekventa ändringar

  %p
    En ändringslogg som endast nämner vissa av ändringarna kan vara lika riskabel
    som att inte ha någon ändringslogg alls. Även om många av ändringarna kanske
    inte är relevanta - till exempel behöver kanske inte borttagningen av ett
    enskilt blanksteg alltid nämnas - bör alla viktiga ändringar nämnas i
    ändringsloggen. Genom att inkonsekvent lägga in ändringar kan dina användare
    felaktigt tro att ändringsloggen är den enda källan till sanning. Så borde
    det vara. Med stor makt följer stort ansvar - att ha en bra ändringslogg
    innebär att ha en konsekvent uppdaterad ändringslogg.

  %aside
    Det var inte allt. Hjälp mig att samla dessa antimönster genom att
    = link_to "skapa en issue", data.links.issue
    eller en pull requests

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Vanliga frågor

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Finns det ett standardformat på ändringsloggar?

  %p
    Inte riktigt. Det finns #{link_to "GNU:s stilguide för ändringsloggar", data.links.gnustyle} och
    den #{link_to "två stycken-långa GNU NEWS-filen", data.links.gnunews} med "riktlinjer".
    Båda är bristfälliga och otillräckliga.

  %p
    Detta projekt har som mål att bli
    = link_to "en bättre konvention för ändringsloggar.", data.links.changelog
    Det utgår från uppenbart goda praxis från öppen källkods-världen och sammanför dem.

  %p
    Konstruktiv kritik, diskussion och förslag till förbättring
    = link_to "är välkommen.", data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Vad bör filen med ändringsloggen heta?

  %p
    Döp den till <code>CHANGELOG.md</code>. En del projekt använder
    <code>HISTORY</code>, <code>NEWS</code> eller <code>RELEASES</code>.

  %p
    Även om det är lätt att tänka att det inte spelar så stor roll vad filen
    med ändringsloggar kallas, varför göra det svårare för dina slutanvändare
    att enkelt hitta de viktigaste ändringarna?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Hur är det med GitHub Releases?

  %p
    Det är ett fantasiskt initiativ. #{link_to "Releases", data.links.github_releases} kan användas
    för att göra enkla git-taggar (t.ex. en tagg kallad <code>v1.0.0</code>)
    till formaterade versionsanteckningar genom att manuellt lägga till
    versionsanteckningar eller så kan den hämta meddelandena i kommenterade
    git-taggar och omvandla dessa till versionsanteckningar.

  %p
    GitHub Releases skapar en icke porterbar ändringslogg som enbart kan visas
    för användare på GitHub. Det är möjligt att formatera det ungefär som på
    För en ändringslogg-formatet, men det tenderar att bli lite mer invecklat.

  %p
    Nuvarande version av GitHub releases är möjligtvis också lite svår att
    hitta för slutanvändare, till skillnad från filer med normalt stora
    bokstäver (<code>README</code>, <code>CONTRIBUTING</code>, etc.).
    Ett annat bekymmer är att användargränssnittet för närvarande inte
    erbjuder länkar till incheckningsloggar mellan olika versioner.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Kan ändringsloggar bli automatiskt tolkade?

  %p
    Det är svårt då människor använder vitt olika format och filnamn.

  %p
    #{link_to "Vandamme", data.links.vandamme} är en Ruby gem skapad av gruppen
    Gemnasium som tolkar många (men inte alla)
    ändringsloggar för öppen källkod.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Hur är det med återtagna utgåvor ("yanked")?

  %p
    Återtagna utgåvor är versioner som måste tas tillbaka på grund av
    allvarliga programfel eller säkerhetsproblem. Oftast brukar dessa versioner
    inte ens finnas med i ändringsloggarna. De borde det. Så här borde du
    visa dem:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    Taggen <code>[YANKED]</code> står ut av en anledning; det är viktigt
    att den syns. Då den är omsluten med hakparenteser är det också lättare
    för ett program att tolka den.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Borde du någonsin förändra en ändringslogg?

  %p
    Självklart. Det finns alltid en bra anledning att förbättra en ändringslogg.
    Jag öppnar regelbundet pull requests för att lägga till saknade utgåvor
    för öppna källkodsprojekt som inte tar hand om sin ändringslogg.

  %p
    Det kan också hända att du upptäcker att du glömt att ta upp en icke
    bakåtkompatibel förändring i en version. I sådana fall är det självklart
    viktigt att uppdatera din ändringslogg.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Hur kan jag bidra?

  %p
    Detta dokument är inte <strong>sanningen</strong>, det är en noga övervägd
    åsikt tillsammans med information och exempel jag har samlat på mig.

  %p
    Detta beror på att jag vill att vår gemenskap ska nå enighet. Jag tror på
    att diskussionen är lika viktig som slutresultatet.

  %p
    Så tveka inte att <strong>#{link_to "kasta dig in i diskussionen", data.links.repo}</strong>.

.press
  %h3 Samtal
  %p
    Jag var med i #{link_to "The Changelog podcast", data.links.thechangelog}
    för att prata om varför förvaltare och bidragsgivare bör bry sig om
    ändringsloggar, och motiveringen bakom detta projekt.
```

## File: source/tr-TR/0.3.0/index.html.haml
```haml
---
description: Arkadaşlarınızın, git kayıtlarını CHANGELOG dosyalarına yığmasını engelleyin
title: Değişiklik kaydı tutun
language: tr-TR
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### Nedir bu değişiklik kayıtları?
  Değişiklik kayıtları bir proje için özel olarak hazırlanmış, tarihsel sıralamayla
  sıralanmış, önemli değişikliklerin bir bütünüdür.

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### Değişikliklerin kayıtlarını tutmanın anlamı ne?
  Bir projenin kullanıcılarının ya da katılımcılarının, dağıtımlar (ya da sürümler)
  arasındaki tam olarak hangi önemli değişikliklerin olduğunu takip edebilmelerini sağlar.

  ### Neden umrumda olsun ki?
  Çünkü yazılım paketleri insanlar içindir. Eğer umrunuzda değilse neden açık kaynağa
  katkıda bulunuyorsunuz ki? Mutlaka sevimli beyninizin bir yerlerinde bunu önemseyen
  bir çekirdek (a-ha!) vardır.

  [Adam Stacoviak ve Jerod Santo ile Changelog Podcast'inde][thechangelog] (uyuyor,
  değil mi?) neden geliştiricilerin ve katılımcıların umrunda olması gerektiğini ve
  bu projenin arkasındaki motivasyonu konuştuk. Eğer vaktiniz varsa (1:06) iyi bir
  söyleşi oldu.

  ### Bir değişiklik kaydını iyi yapan nedir?
  Bunu sorduğunuza sevindim.

  İyi bir değişiklik kaydı şu prensiplere bağlıdır:

  - İnsanlar için yapılmıştır, makineler için değil, yani okunabilirliği kritiktir.
  - Kolayca bölümler arası bağlantı kurulabilmelidir. (Bu yüzden yalın metin yerine markdown)
  - Her sürüm için bir alt bölüm içermelidir.
  - Dağıtımları tersine tarih sırası ile listemelidir. (En yeni en üstte)
  - Tüm tarihler `YYYY-AA-GG` biçiminde olmalıdır. (Örneğin `2 Haziran 2012` için `2012-06-02`) Uluslararasıdır, [anlamlıdır](https://xkcd.com/1179/), ve lisan bağımsızdır.
  - [Anlamsal sürümleme][semver]nin desteklenip desteklenmediğini özellikle belirtilmelidir.
  - Her sürümde olması gerekenler:
    - Üstteki biçimde dağıtım tarihi.
    - Projeye etkileri bakımından değişikliklerin gruplanması, şöyle ki:
      - `Eklendi`: yeni özellikler için,
      - `Değişti`: var olan beceriler değiştiyse,
      - `Rafa kalktı`: gelecekte yok olacak var olan beceriler,
      - `Kaldırıldı`: bu sürümde kaldırılan, daha önce rafa kaldırılmış olan beceriler,
      - `Düzeltildi`: ayıklanmış hatalar,
      - `Güvenlik`: açıkları kapatabilmeleri için kullanıcıları bilgilendirin.

  ### Gerekli çabayı nasıl en aza indirebilirim?
  Her zaman en üstte değişiklikleri takip ettiğiniz bir `"Yayımlanmadı"` bölümü olsun.

  Bu, iki amaca hizmet eder:

  - İnsanlar gelecek sürümlerde karşılarına ne gibi değişiklikler çıkacağını görebilirler
  - Dağıtım zamanı geldiğinde `"Yayımlanmadı"` başlığını sürüm numarası ile değişitip
    tepeye yeni bir `"Yayımlanmadı"` bölümü eklemeniz yeterli.

  ### Tek boynuzlu atları ağlatan ne?
  Peki… Gelelim sadede.

  - **Commit kayıtlarının farkını yüklemek.** Sakın bunu yapmayın, kimseye yardım etmiyorsunuz.
  - **Rafa kaldırılanları ön plana çıkarmamak.** Kullanıcılar için bir sürümden diğerine
    yükseltme yaptıklarında neyin bozulabileceği apaçık ortada olmalı.
  - **Bölgesel biçimlenmiş tarihler.** ABD'de insanlar ay bilgisini önce kullanıyor
    ("06-02-2012" demek 2 Haziran 2012 demek yani, ki tamamen *saçma*), diğer taraftan
    dünyanın bir çok bölgesinde "2 Haziran 2012" gibi robotik bir yapı kullanıp farklı
    şekilde dile getiriyor. "2012-06-02" hem mantıksal olarak en büyüğünden en küçüğüne çalışıyor,
    hem de diğer tarih biçimleriyle çakışmıyor. Aynı zamanda [ISO standardında](http://www.iso.org/iso/home/standards/iso8601.htm).
    Bu yüzden değişiklik kayıtları için önerilen tarih biçimidir.

  Dahası da var. Bu tek boynuzlu at gözyaşlarını toplamak için
  [bir başlık açın][issues]
  ya da bir çekme isteği (pull request) gönderin.

  ### Standart bir değişiklik kaydı biçimi var mı?
  Ne yazık ki hayır. Sakin olun. Biliyorum, şu an alelacele GNU değişiklik kaydı
  stil rehberi için bağlantı arıyorsunuz, ya da iki paragraflık GNU NEWS dosyasına
  bakınıyorsunuz. GNU stil rehberi iyi bir başlangıç fakat üzücü derece naif.
  Naif olmakta yanlış bir şey yok, fakat insanlar rehberlik aradığında nadiren
  yardımcı oluyorlar. Özellikle ortada uğraşılacak bir çok durum ve uç örnekler
  mevcutken.

  Bu proje [umuyorum ki daha iyi CHANGELOG dosyaları için bir altyapı oluşturacak][CHANGELOG].
  Mevcut durumun çok iyi olduğunu düşünmüyorum, ve topluluk olarak, gerçek yazılım
  projelerinden iyi pratikleri toplayarak bundan daha iyisini yapabiliriz. Lütfen
  etrafa bir göz gezdirin ve unutmayın [gelişme yolunda tartışmalara ve önerilere her zaman açığız][issues]!

  ### Değişiklik kayıtlarının dosya ismi ne olmalı?
  Eh, üstteki örnekten çıkartamadıysanız eğer, `CHANGELOG.md` şu ana kadarki
  en iyi çözüm.

  Bazı projeler `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`, `NEWS.md`,
  `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md` vb de kullanıyor.

  Tam bir karmaşa. Tüm bu isimler insanların bu dosyayı bulmasını zorlaştırıyor.

  ### Neden `git log` farkları kullanılmasın?
  Çünkü kayıt farkları bir sürü parazit içerir - doğal olarak. Mükemmel insanlar
  tarafından yürütülen, hiç yazım hatasının yapılmadığı, dosyaların gönderilmesinin
  hiç unutulmadığı, refaktör yapılmasının atlanmadığı varsayımsal bir projede bile,
  uygun bir değişiklik kaydı oluşmayacaktır. Dosyaları repoya göndermenin amacı,
  atomik seviyede kodun bir durumdan diğerine geçişinin sağlanmasıdır. Değişiklik
  kayıtları ise, bu durumlar arasında önem arz eden değişiklikleri belgelemeyi
  amaçlıyor.

  İyi yorumlar ve kodun kendisi arasındaki fark,
  aynı şekilde değişiklik kayıtları ve commit kayıtları arasındaki gibidir:
  biri *neden* olduğunu, diğeri nasıl olduğunu açıklar.

  ### Değişiklik kayıtları otomatik olarak toplanabilir mi?
  Zor, çünkü insanlar bir çok farklı biçim ve dosya isimleri kullanıyorlar.

  [Vandamme][vandamme], [Gemnasium][gemnasium] ekibi tarafından oluşturulmuş
  bir Ruby Gem'idir ve bir çok (hepsini değil) açık kaynaklı projenin değişiklik
  kayıtlarını ayrıştırabiliyor.

  ### Neden arada bir "CHANGELOG" ve arada bir "değişiklik kaydı" yazıyorsun?
  "CHANGELOG" dosyanın ismi. Biraz fazla şaşalı ama bir çok açık kaynak kodlu
  proje tarafından uygulanan tarihi bir gelenek. Benzer dosyalar da var;
  [`README`][README], [`LICENSE`][LICENSE], ve [`CONTRIBUTING`][CONTRIBUTING].

  Büyük harfle isimlendirme (eski işletim sistemlerinde dosyaların tepede
  gözükmelerini sağlardı) dikkat çekmek için. Proje hakkında önemli meta verileri
  içerdikleri için, kullanmak ya da katkıda bulunmak isteyen herkesin işine
  yarar, tıpkı [açık kaynaklı proje rozetleri][shields] gibi.

  "Değişiklik kaydı"ndan bahsettiğim zamanlar bu dosyanın işlevinden bahsediyorum:
  Değişiklikleri kaydetmek.

  ### Peki ya Geri çekilen dağıtımlar?
  Geri çekilen dağıtımlar, önemli hatalar ya da güvenlik sebepleri nedeniyle
  yayından geri çekilen sürümlerdir. Genelde bu sürümler değişiklik kayıtlarında
  görüntülenmezler. Görünmeliler. Tam da şu şekilde görünmeliler:

  `## 0.0.5 - 2014-12-13 [GERİ ÇEKİLDİ]`

  `[GERİ ÇEKİLDİ]` etiketi belirli bir sebepten büyük harf. İnsanların bunu fark
  etmeleri çok önemli. Ayrıca köşeli parantezler ile çevrelenmiş olması
  programatik olarak da ayrıştırılabilmesine olanak sağlıyor.

  ### Değişiklik kayıtlarınızı tekrar yazmalı mısınız?
  Tabii ki. Her zaman değişiklik kayıtlarını geliştirmek için iyi sebepler vardır.
  Düzenli olarak açık kaynaklı projelerde bakım yapılmayan değişiklik kayıtları
  için çekme istekleri yapıyorum.

  Ayrıca bir sürümdeki notların arasında önemli bir değişiklikten bahsetmeyi
  unutmuş olduğunuzu fark edebilirsiniz. Değişiklik kayıtlarınızı bu bilgi ışığında
  güncellemeniz gerektiği gün gibi ortada.

  ### Nasıl katkıda bulunabilirim?
  Bu belge **doğrunun kendisi** değil; benim hesaplı görüşlerimdir. Beraberinde
  toparlamış olduğum bilgiler ve örnekler bulunur.
  [GitHub repo][gh]sunda güncel bir [CHANGELOG][] sağlıyor olsam da, özellikle
  bir *sürüm* ya da ([SemVer.org][semver] benzeri) takip edilecek kurallar
  oluşturmadım.

  Bunu istememin sebebi topluluğun ortak bir paydada buluşmasını istememdir.
  İnanıyorum ki tartışmanın kendisi de sonucu kadar önemli.

  Yani lütfen, [**siz de katılın**][gh].

  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/tr-TR/1.0.0/index.html.haml
```haml
---
title: Değişiklik kaydı tutun
description: Arkadaşlarınızın, git mesajlarını değişiklik kayıtlarına yığmasını engelleyin.
language: tr-TR
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Nedir bu değişiklik kayıtları?

  %p
    Değişiklik kayıtları bir proje için özel olarak hazırlanmış,
    tarihsel sıralamayla sıralanmış, önemli değişikliklerin bir bütünüdür.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Değişikliklerin kayıtlarını tutmanın anlamı ne?

  %p
    Bir projenin kullanıcılarının ya da katılımcılarının, dağıtımlar
    (ya da sürümler) arasındaki tam olarak hangi önemli değişikliklerin
    olduğunu takip edebilmelerini sağlar.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kim değişiklik kayıtlarına ihtiyaç duyar ki?

  %p
    İnsanlar. İster tüketici olsun, ister geliştirici, kullanılan yazılımın
    son kullanıcıları, o yazılımın içinde ne olduğunu önemseyen kişilerdir.
    Yazılım değiştiğinde, insanlar neden ve nasıl olduğunu bilmek isterler.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Nası iyi değişiklik kayıtları tutarım?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Rehber prensipler

  %ul
    %li
      Değişiklik kayıtları <em>insanlar</em> içindir, makineler için değil.
    %li
      Her sürüm için bir girdi içermelidir.
    %li
      Benzer değişiklikler gruplanmalıdır.
    %li
      Sürümler ve bölümlere bağlantı verilebilir olmalıdır.
    %li
      En son sürüm ilk başta olmalıdır.
    %li
      Her sürümün dağıtım tarihi bulunmalıdır.
    %li
      Geliştirirken #{link_to "anlamlı sürümlendirme (Semver)", data.links.semver} kullanıp kullanmadığınızı bildirin.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Değişiklik tipleri

  %ul
    %li
      %code Eklendi
      \: Yeni özellikler için.
    %li
      %code Değişti
      \: Var olan becerilerde yapılan değişiklikler için.
    %li
      %code Rafa kalktı
      \: Gelecekte yok olacak beceriler için.
    %li
      %code Kaldırıldı
      \: Kaldırılan beceriler için.
    %li
      %code Düzeltildi
      \: Ayıklanmış hatalar için.
    %li
      %code Güvenlik
      \: Bir güvenlik açığı söz konusuysa.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Gerekli çabayı nasıl en aza indirebilirim?

  %p
    Her zaman en üstte, değişiklikleri takip ettiğiniz bir <code>Yayımlanmadı</code>
    bölümü olsun

  %p Bu, iki amaca hizmet eder:

  %ul
    %li
      İnsanlar gelecek sürümlerde karşılarına ne gibi değişiklikler çıkacağını görebilirler
    %li
      Dağıtım zamanı geldiğinde <code>Yayımlanmadı</code> bölümünü
      yeni dağıtım sürümü bölümü olarak kullanabilirsiniz.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Değişiklik kütükleri kötü olabilirler mi?

  %p Evet. Buyrun size işe yaramayacak bir kaç örnek;

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit kayıtlarının farkları

  %p
    Değişiklik kayıtları için commit kayıtlarının farklarını kullanmak
    kötü bir fikirdir: genellikle çok gürültülü olurlar. Commit birleşmeleri,
    kötü başlıklı commitler, belgeleme değişiklikleri vb.

  %p
    Bir commit yapılmasının sebebi, kodun bir sonraki aşamaya evrilmesidir.
    Bazı projeler commitleri temizler, bazıları temizlemez.

  %p
    Değişiklik kayıtlarına eklenen bir girdi ise, öneme sahip bir değişikliğin
    belgelenmesi amaçlıdır. Genelde bir çok commit işlemini kapsar ve son
    kullanıcıyla iletişimi açık tutar.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Rafa kalkanları göz ardı etmek

  %p
    İnsanlar bir sürümden diğerine yükselttiklerinde, bir şeylerin bozulup
    bozulmayacağı acı verici derecede açık olmalıdır. Rafa kalkan özelliklerin
    listelendiği sürüme geçip, bu rafa kaldırılanlara yönelik kendi geliştirmelerini
    yaparak, en nihayetinde özelliklerin tamamen kaldırıldığı sürüme
    geçiş yapabilmeliler.

  %p
    Eğer hiç bir şey yapmasanız bile, rafa kalkanları, kaldırılanları ve
    önemli değişiklikleri, değişiklik kayıtlarınızda listeleyin.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Kafa karıştırıcı tarihler

  %p
    A.B.D.'de insanlar ay kısmını önce kullanırken (2 Haziran 2012 için
    <code>06-02-2012</code>), dünyanın bir çok bölümünde daha robotik bir
    kullanım <code>2 Haziran 2012</code> söz konusu. <code>2012-06-02</code>
    biçimi en küçüğünden en büyüğüne tüm biçimlerle çakışmadan kullanılabiliyor
    ve aynı zamanda bir #{link_to "ISO standardı", data.links.isodate}. Bu sebeple değişiklik
    kayıtları için önerilen tarih biçimidir.

  %aside
    Mutlaka dahası da vardır. Benzer durumları toplamam için
    = link_to "bir çağrı açın", data.links.issue
    ya da bir çekme isteği gönderin.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Sıkça sorulan sorular

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Standart bir değişiklik kayıt biçimi var mı?

  %p
    Pek sayılmaz. GNU değişiklik kayıtları stil rehberi mevcut ya da
    iki paragraflık GNU NEWS "rehber" dosyası var. İkisi de uygun
    değiller ve yetersizler.

  %p
    Bu proje daha iyi
    = link_to "bir değişiklik kayıtları düzeni", data.links.changelog
    oluşturmaya çalışıyor. Bunun için de açık kaynaklı topluluklardaki
    en iyi kullanımları inceleyip, topluyoruz.

  %p
    Sağlıklı eleştiriler, tartışmalar ve öneriler, projenin gelişmesi
    için her zaman
    = link_to "hoş karşılanır.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Değişiklik kayıtları dosyasının ismi ne olmalı?

  %p
    İsterseniz <code>CHANGELOG.md</code> olarak isimlendirin. Bazı projeler
    <code>HISTORY</code>, <code>NEWS</code> ya da <code>RELEASES</code>
    kullanıyor.

  %p
    Dosya isminin çok da önemli olmadığını düşünebilirsiniz, fakat
    neden kullanıcılarınızın değişiklikleri takip edebilmesi için
    onların işlerini zorlaştırasınız ki?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Peki ya GitHub dağıtımları?

  %p
    Harika bir girişim. #{link_to "Dağıtımlar", data.links.github_releases} içine kendiniz
    değişiklik kayıtları eklerseniz basit git etiketlerini
    (örneğin <code>v1.0.0</code>) zengin dağıtım notlarına çevirebilir
    ya da notlar eklenmiş git etiketlerinden oluşturulabilirsiniz.

  %p
    GtHub dağıtımları sadece GitHub içeriğinde görüntülenebilecek,
    taşınamaz değişiklik kayıtları oluşturur. Biraz emek harcayarak
    "Değişiklik kayıtları tutun" biçimine uygun hale getirilebilir.

  %p
    Ayrıca GitHub dağıtımlarının şu anki hali son kullanıcılar tarafından
    çok kolay bulunabilir değil. Tipik büyük harfli dosyalar
    (<code>README</code>, <code>CONTRIBUTING</code>, vb.) daha çok göze
    çarpıyor. Bir başka konu da, mevcut arayüz her dağıtım arasındaki
    commit kayıtlarına bağlantı vermeye izin vermiyor..

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Değişiklik kayıtları otomatik olarak toplanabilir mi?

  %p
    Zor, çünkü insanlar bir çok farklı biçim ve dosya isimleri
    kullanıyorlar.

  %p
    #{link_to "Vandamme", data.links.vandamme}, Gemnasium
    ekibi tarafından oluşturulmuş bir Ruby Gem'i ve bir çok (ama hepsi
    değil) açık kaynak projenin değişiklik kayıtlarını okuyabiliyor.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Peki ya geri çekilen dağıtımlar?

  %p
    Geri çekilen dağıtımlar, önemli hatalar ya da güvenlik sebepleri nedeniyle
    yayından geri çekilen sürümlerdir. Genelde bu sürümler değişiklik kayıtlarında
    görüntülenmezler. Görünmeliler. Tam da şu şekilde görünmeliler:

  %p <code>## 0.0.5 - 2014-12-13 [GERİ ÇEKİLDİ]</code>

  %p
    <code>[GERİ ÇEKİLDİ]</code> etiketi belirli bir sebepten büyük harf.
    İnsanların bunu fark etmeleri çok önemli. Ayrıca köşeli parantezler
    ile çevrelenmiş olması programatik olarak da ayrıştırılabilmesine
    olanak sağlıyor.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Değişiklik kayıtlarınızı tekrar yazmalı mısınız?

  %p
    Tabii ki. Her zaman değişiklik kayıtlarını geliştirmek için iyi sebepler vardır.
    Düzenli olarak açık kaynaklı projelerde bakım yapılmayan değişiklik kayıtları
    için çekme istekleri yapıyorum.

  %p
    Ayrıca bir sürümdeki notların arasında önemli bir değişiklikten bahsetmeyi
    unutmuş olduğunuzu fark edebilirsiniz. Değişiklik kayıtlarınızı bu bilgi ışığında
    güncellemeniz gerektiği gün gibi ortada.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Nasıl katkıda bulunabilirim?

  %p
    Bu belge <strong>doğrunun kendisi</strong> değil; benim ince eleyip
    sık dokuduğum görüşlerimdir. Beraberinde toparlamış olduğum bilgiler
    ve örnekler bulunur.

  %p
    Burada yapmaya çalıştığım topluluğun ortak bir paydada buluşmasını sağlamak.
    İnanıyorum ki tartışmanın kendisi de sonucu kadar önemli.

  %p
    Yani lütfen, <strong>#{link_to "siz de katılın", data.links.repo}</strong>.

.press
  %h3 Sohbetler
  %p
    Geliştiricilerin ve katkıda bulunanların neden değişiklik kayıtlarını
    dikkate almaları gerekliliğini ve bu projenin arkasındaki motivasyonu
    anlattığım #{link_to "Değişiklik Kayıtları podcast", data.links.thechangelog}'ini
    inceleyebilirsiniz.
```

## File: source/tr-TR/1.1.0/index.html.haml
```haml
---
title: Değişiklik kaydı tutun
description: Arkadaşlarınızın, git mesajlarını değişiklik kayıtlarına yığmasını engelleyin.
language: tr-TR
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Nedir bu değişiklik kayıtları?

  %p
    Değişiklik kayıtları bir proje için özel olarak hazırlanmış,
    tarihsel sıralamayla sıralanmış, önemli değişikliklerin bir bütünüdür.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Değişikliklerin kayıtlarını tutmanın anlamı ne?

  %p
    Bir projenin kullanıcılarının ya da katılımcılarının, dağıtımlar
    (ya da sürümler) arasındaki tam olarak hangi önemli değişikliklerin
    olduğunu takip edebilmelerini sağlar.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Kim değişiklik kayıtlarına ihtiyaç duyar ki?

  %p
    İnsanlar. İster tüketici olsun, ister geliştirici, kullanılan yazılımın
    son kullanıcıları, o yazılımın içinde ne olduğunu önemseyen kişilerdir.
    Yazılım değiştiğinde, insanlar neden ve nasıl olduğunu bilmek isterler.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Nası iyi değişiklik kayıtları tutarım?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Rehber prensipler

  %ul
    %li
      Değişiklik kayıtları <em>insanlar</em> içindir, makineler için değil.
    %li
      Her sürüm için bir girdi içermelidir.
    %li
      Benzer değişiklikler gruplanmalıdır.
    %li
      Sürümler ve bölümlere bağlantı verilebilir olmalıdır.
    %li
      En son sürüm ilk başta olmalıdır.
    %li
      Her sürümün dağıtım tarihi bulunmalıdır.
    %li
      Geliştirirken #{link_to "anlamlı sürümlendirme (Semver)", data.links.semver} kullanıp kullanmadığınızı bildirin.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Değişiklik tipleri

  %ul
    %li
      %code Eklendi
      \: Yeni özellikler için.
    %li
      %code Değişti
      \: Var olan becerilerde yapılan değişiklikler için.
    %li
      %code Rafa kalktı
      \: Gelecekte yok olacak beceriler için.
    %li
      %code Kaldırıldı
      \: Kaldırılan beceriler için.
    %li
      %code Düzeltildi
      \: Ayıklanmış hatalar için.
    %li
      %code Güvenlik
      \: Bir güvenlik açığı söz konusuysa.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Gerekli çabayı nasıl en aza indirebilirim?

  %p
    Her zaman en üstte, değişiklikleri takip ettiğiniz bir <code>Yayımlanmadı</code>
    bölümü olsun

  %p Bu, iki amaca hizmet eder:

  %ul
    %li
      İnsanlar gelecek sürümlerde karşılarına ne gibi değişiklikler çıkacağını görebilirler
    %li
      Dağıtım zamanı geldiğinde <code>Yayımlanmadı</code> bölümünü
      yeni dağıtım sürümü bölümü olarak kullanabilirsiniz.

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Değişiklik kütükleri kötü olabilirler mi?

  %p Evet. Buyrun size işe yaramayacak bir kaç örnek;

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Commit kayıtlarının farkları

  %p
    Değişiklik kayıtları için commit kayıtlarının farklarını kullanmak
    kötü bir fikirdir: genellikle çok gürültülü olurlar. Commit birleşmeleri,
    kötü başlıklı commitler, belgeleme değişiklikleri vb.

  %p
    Bir commit yapılmasının sebebi, kodun bir sonraki aşamaya evrilmesidir.
    Bazı projeler commitleri temizler, bazıları temizlemez.

  %p
    Değişiklik kayıtlarına eklenen bir girdi ise, öneme sahip bir değişikliğin
    belgelenmesi amaçlıdır. Genelde bir çok commit işlemini kapsar ve son
    kullanıcıyla iletişimi açık tutar.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Rafa kalkanları göz ardı etmek

  %p
    İnsanlar bir sürümden diğerine yükselttiklerinde, bir şeylerin bozulup
    bozulmayacağı acı verici derecede açık olmalıdır. Rafa kalkan özelliklerin
    listelendiği sürüme geçip, bu rafa kaldırılanlara yönelik kendi geliştirmelerini
    yaparak, en nihayetinde özelliklerin tamamen kaldırıldığı sürüme
    geçiş yapabilmeliler.

  %p
    Eğer hiç bir şey yapmasanız bile, rafa kalkanları, kaldırılanları ve
    önemli değişiklikleri, değişiklik kayıtlarınızda listeleyin.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Kafa karıştırıcı tarihler

  %p
    A.B.D.'de insanlar ay kısmını önce kullanırken (2 Haziran 2012 için
    <code>06-02-2012</code>), dünyanın bir çok bölümünde daha robotik bir
    kullanım <code>2 Haziran 2012</code> söz konusu. <code>2012-06-02</code>
    biçimi en küçüğünden en büyüğüne tüm biçimlerle çakışmadan kullanılabiliyor
    ve aynı zamanda bir #{link_to "ISO standardı", data.links.isodate}. Bu sebeple değişiklik
    kayıtları için önerilen tarih biçimidir.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Tutarsız değişiklikler

  %p
    Sadece bazı değişiklikleri içeren bir değişiklik kütüğü en az hiç 
    olmaması kadar tehlikelidir. Bir çok değişikliğin kayıt altına alınması
    gerekmese bile - örneğin tek bir boşluğun kaldırılmasının kayıt altına
    alınması gerekmeyebilir - her türlü önemli değişiklikten kayıt kütüğünde
    bahsedilmelidir. Tutarsız bir şekilde değişiklikleri uygulamak,
    kullanıcıların tek doğrunun sadece değişiklik kütüğünde var olanlar
    olduğunu sanmasına yol açabilir. Öyle de olmalı. Büyük güç beraberinde
    büyük sorumluluk getirir - iyi değişiklik kayıtları demek tutarlı bir
    şekilde güncellenen değişiklik kayıtları demektir.

  %aside
    Mutlaka dahası da vardır. Benzer durumları toplamam için
    = link_to "bir çağrı açın", data.links.issue
    ya da bir çekme isteği gönderin.

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Sıkça sorulan sorular

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Standart bir değişiklik kayıt biçimi var mı?

  %p
    Pek sayılmaz. GNU değişiklik kayıtları stil rehberi mevcut ya da
    iki paragraflık GNU NEWS "rehber" dosyası var. İkisi de uygun
    değiller ve yetersizler.

  %p
    Bu proje daha iyi
    = link_to "bir değişiklik kayıtları düzeni", data.links.changelog
    oluşturmaya çalışıyor. Bunun için de açık kaynaklı topluluklardaki
    en iyi kullanımları inceleyip, topluyoruz.

  %p
    Sağlıklı eleştiriler, tartışmalar ve öneriler, projenin gelişmesi
    için her zaman
    = link_to "hoş karşılanır.", data.links.issue


  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Değişiklik kayıtları dosyasının ismi ne olmalı?

  %p
    İsterseniz <code>CHANGELOG.md</code> olarak isimlendirin. Bazı projeler
    <code>HISTORY</code>, <code>NEWS</code> ya da <code>RELEASES</code>
    kullanıyor.

  %p
    Dosya isminin çok da önemli olmadığını düşünebilirsiniz, fakat
    neden kullanıcılarınızın değişiklikleri takip edebilmesi için
    onların işlerini zorlaştırasınız ki?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Peki ya GitHub dağıtımları?

  %p
    Harika bir girişim. #{link_to "Dağıtımlar", data.links.github_releases} içine kendiniz
    değişiklik kayıtları eklerseniz basit git etiketlerini
    (örneğin <code>v1.0.0</code>) zengin dağıtım notlarına çevirebilir
    ya da notlar eklenmiş git etiketlerinden oluşturulabilirsiniz.

  %p
    GtHub dağıtımları sadece GitHub içeriğinde görüntülenebilecek,
    taşınamaz değişiklik kayıtları oluşturur. Biraz emek harcayarak
    "Değişiklik kayıtları tutun" biçimine uygun hale getirilebilir.

  %p
    Ayrıca GitHub dağıtımlarının şu anki hali son kullanıcılar tarafından
    çok kolay bulunabilir değil. Tipik büyük harfli dosyalar
    (<code>README</code>, <code>CONTRIBUTING</code>, vb.) daha çok göze
    çarpıyor. Bir başka konu da, mevcut arayüz her dağıtım arasındaki
    commit kayıtlarına bağlantı vermeye izin vermiyor..

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Değişiklik kayıtları otomatik olarak toplanabilir mi?

  %p
    Zor, çünkü insanlar bir çok farklı biçim ve dosya isimleri
    kullanıyorlar.

  %p
    #{link_to "Vandamme", data.links.vandamme}, Gemnasium
    ekibi tarafından oluşturulmuş bir Ruby Gem'i ve bir çok (ama hepsi
    değil) açık kaynak projenin değişiklik kayıtlarını okuyabiliyor.


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    Peki ya geri çekilen dağıtımlar?

  %p
    Geri çekilen dağıtımlar, önemli hatalar ya da güvenlik sebepleri nedeniyle
    yayından geri çekilen sürümlerdir. Genelde bu sürümler değişiklik kayıtlarında
    görüntülenmezler. Görünmeliler. Tam da şu şekilde görünmeliler:

  %p <code>## 0.0.5 - 2014-12-13 [GERİ ÇEKİLDİ]</code>

  %p
    <code>[GERİ ÇEKİLDİ]</code> etiketi belirli bir sebepten büyük harf.
    İnsanların bunu fark etmeleri çok önemli. Ayrıca köşeli parantezler
    ile çevrelenmiş olması programatik olarak da ayrıştırılabilmesine
    olanak sağlıyor.

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Değişiklik kayıtlarınızı tekrar yazmalı mısınız?

  %p
    Tabii ki. Her zaman değişiklik kayıtlarını geliştirmek için iyi sebepler vardır.
    Düzenli olarak açık kaynaklı projelerde bakım yapılmayan değişiklik kayıtları
    için çekme istekleri yapıyorum.

  %p
    Ayrıca bir sürümdeki notların arasında önemli bir değişiklikten bahsetmeyi
    unutmuş olduğunuzu fark edebilirsiniz. Değişiklik kayıtlarınızı bu bilgi ışığında
    güncellemeniz gerektiği gün gibi ortada.

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Nasıl katkıda bulunabilirim?

  %p
    Bu belge <strong>doğrunun kendisi</strong> değil; benim ince eleyip
    sık dokuduğum görüşlerimdir. Beraberinde toparlamış olduğum bilgiler
    ve örnekler bulunur.

  %p
    Burada yapmaya çalıştığım topluluğun ortak bir paydada buluşmasını sağlamak.
    İnanıyorum ki tartışmanın kendisi de sonucu kadar önemli.

  %p
    Yani lütfen, <strong>#{link_to "siz de katılın", data.links.repo}</strong>.

.press
  %h3 Sohbetler
  %p
    Geliştiricilerin ve katkıda bulunanların neden değişiklik kayıtlarını
    dikkate almaları gerekliliğini ve bu projenin arkasındaki motivasyonu
    anlattığım #{link_to "Değişiklik Kayıtları podcast", data.links.thechangelog}'ini
    inceleyebilirsiniz.
```

## File: source/uk/1.0.0/index.html.haml
```haml
---
title: Ведіть Changelog
description: Не дозволяйте друзям зливати логи гіта у лог змін.
language: uk
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Що таке лог змін?

  %p
    Лог змін - це файл, що містить підтримуваний та хронологічно
    впорядкований список змін для кожної версії проекту.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Навіщо вести лог змін?

  %p
    Це дозволяє полегшити користувачам та контриб'юторам слідкувати за змінами
    у релізах (чи версіях) проекту.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Кому потрібен лог змін?

  %p
    Людям. Користувачам та розробникам, кінцевими користувачами програмного
    забезпечення є люди, яким важливо знати з чим вони працюють.
    Якщо відбулися змін, то люди повинні знати що змінилося і як.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Як створити хороший лог змін?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Головні принципи

  %ul
    %li
      Лог змін <em>для людей</em>, а не машин.
    %li
      Окремий розділ для кожної версії.
    %li
      Зміни одного типу мають бути згруповані.
    %li
      На версії та секції потрібно ставити гіперпосилання.
    %li
      Остання версія мусить бути на початку.
    %li
      Кожна версія має мати власну дату.
    %li
      Вкажіть чи підтримуєте Ви принципи #{link_to "Семантичне версіювання", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Типи змін

  %ul
    %li
      %code Додано
      для нового функціоналу.
    %li
      %code Змінено
      для змін в існуючий функціонал.
    %li
      %code Застаріло
      для функціоналу, що планується видалити.
    %li
      %code Видалено
      про вже видалений функціонал.
    %li
      %code Виправлення
      для будь яких виправлень.
    %li
      %code Безпека
      при виявленні вразливостей.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Як мені докладати мінімальні зусилля для ведення логу змін?

  %p
    Ведіть розділ <code>Нове</code> на початку файла.

  %p Для переслідування подвійної цілі:

  %ul
    %li
      Люди можуть бачити майбутні зміни в найближчому релізі
    %li
      Як настане час релізу, Ви можете перемістити розділ <code>Нове</code>
      у розділ нового релізу,

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Чи може лог змін бути поганим?

  %p Так. Ось декілька способів зробити лог змін не вдалим.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Логи змін між комітами.

  %p
    Використовувати логи комітів як логи змін - це погана ідея.
    Вони наповнені інформаційним шумом. Такими як коміти злиття,
    не інформативні назви комітів, змінами у документації тощо.

  %p
    Призначення комітів у тому, щоб документувати кроки в еволюції коду.
    У деяких проектах історія комітів доглянута, в інших - ні.

  %p
    Призначення ж логу змін полягає у документації вагомих змін,
    часто між багатьма комітами, доносячи їх призначення до кінцевого
    користувача.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ігнорування застарілого функціоналу

  %p
    Коли люди оновлюються з версії до версії, їм потрібна повна впевненість
    у тому, чи може щось зламатися. Їм повинна бути надана можливість оновитися
    до версії зі списком застарілого функціоналу, видалити все застаріле,
    а потім оновитися до версії з видаленим застарілим функціоналом.

  %p
    Якщо Ви не займаєтеся логом змін, то хоча б ведіть список
    застарілого, видаленого або серйозних змін функціоналу.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Незрозумілі дати

  %p
    Регіональні дати можуть відрізнятися і це може бути складно для
    правильного розуміння ваших дат для користувачів у різних куточках
    світу. Варто надавати переваги датам, що форматовані за таким зразком:
    <code>2017-07-17</code>. У них числа ідуть від найбільшого до
    найменшого: рік, місяць, день. Мінімізує конфузи у випадку використання
    регіональних форматів, коли день і місяць можуть мати різний порядок.
    Цей формат не пересікається з більшістю інших форматів і є
    #{link_to "стандартом ISO", data.links.isodate}. Тому такий формат рекомендується для
    логів змін.

  %aside
    Є також інші. Допоможіть зібрати антипатерни,
    = link_to "створіть тікет", data.links.issue
    або пул реквест

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Поширені запитання

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Чи існує стандарт логів змін?

  %p
    Не зовсім. Є   #{link_to "стайлгайд логів змін GNU", data.links.gnustyle}
    або  #{link_to "довгий у два параграфа GNU NEWS file", data.links.gnunews}.
    Обидва неадекватні і неповні.

  %p
    Цей проект покликаний бути
    = link_to "поліпшеною угодою про логи змін.", data.links.changelog
    Це виходить із ліпших практик open source спільноти.

  %p
    Здорова критика, дискусія та пропозиції поліпшення
    = link_to "вітаються.", data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Як назвати файл логів змін?

  %p
    Назвіть <code>CHANGELOG.md</code>. У деяких проектах файл носить назви
    <code>HISTORY</code>, <code>NEWS</code> або <code>RELEASES</code>.

  %p
    Може видатися, що назва файлу для логів змін не суттєва, проте
    навіщо ускладнювати життя користувачам змушуючи їх шукати?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Як щодо GitHub релізів?

  %p
    Це чудова ініціатива. #{link_to "Релізи", data.links.github_releases} можуть бути використані для
    перетворення простих тегів у Git (<code>v1.0.0</code> - до прикладу)
    у деталізовані нотатки до релізів шляхом їх редагування вручну або
    за допомогою коментарів до цих тегів.

  %p
    Релізи GitHub є не портативним логом змін, який може бути показаний
    користувачам лише на самому сайті GitHub. Його можна вести подібно
    до формату Keep a Changelog, але для цього потрібні значні зусилля.

  %p
    Поточна версія на GitHub не так добре очевидна для користувача,
    на відмінну від звичайних файлів з іменами у верхньому регістрі
    (<code>README</code>, <code>CONTRIBUTING</code> і так далі).
    Крім того, інтерфейс не дозволяє посилання на логи комітів між
    релізами.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Чи можуть логи змін автоматично парситися?

  %p
    Це заскладно, оскільки люди використовують різні формати та
    імена файлів.

  %p
    #{link_to "Vandamme", data.links.vandamme} - це гем для Ruby, створений
    командою Gemnasium, який парсить багато (але не всі) логи
    змін проектів з відкритим кодом.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    А як щодо yanked-релізів?

  %p
    Yanked-релізи - це версії, що вилучаються із-за серйозних багів або проблем
    з безпекою у них. Часто вони навіть не згадуються у логах змін.
    А повинні. І описуватися вони повинні так:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> навмисне кидається в очі. Важливо, щоб його
    помітили. Обмежений квадратними дужками, щоб його легше було
    спарсити.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Чи є сенс переписати лог змін?

  %p
    Звісно. Завжди є сенс поліпшувати лог змін. І відкривати пул-реквести
    додаючи втрачені релізи у проекти з відкритим кодом і закинутими
    логами змін.

  %p
    Також можлива ситуація, що Ви забули вказати критичні зміни для версії.
    Очевидно, що потрібно такий лог оновити.



  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Як я можу допомогти вашому проекту?

  %p
    Цей документ не претендує на виключну <strong>правду</strong>;
    Це моє бачення, зі зібраною інформацію та прикладами.

  %p
    Хотів би, щоб спільнота дійшла згоди. Я вірю у дискусію та результат.

  %p
    Тому <strong>#{link_to "беріть участь", data.links.repo}</strong>.

.press
  %h3 Обговорення
  %p
    Я приходив на #{link_to "підкаст The Changelog", data.links.thechangelog}, щоб обговорити
    те, чому ментейнери та контриб'ютори повинні вести логи змін, а також
    про мою мотивацію для створення цього проекту.
```

## File: source/uk/1.1.0/index.html.haml
```haml
---
title: Ведіть Changelog
description: Не дозволяйте друзям зливати логи гіта у лог змін.
language: uk
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2 Не дозволяйте друзям зливати логи гіта у лог змін.

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    Що таке лог змін?

  %p
    Лог змін - це файл, що містить підтримуваний та хронологічно
    впорядкований список змін для кожної версії проекту.

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    Навіщо вести лог змін?

  %p
    Це дозволяє полегшити користувачам та контриб'юторам слідкувати за змінами
    у релізах (чи версіях) проекту.

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    Кому потрібен лог змін?

  %p
    Людям. Користувачам та розробникам, кінцевими користувачами програмного
    забезпечення є люди, яким важливо знати з чим вони працюють.
    Якщо відбулися змін, то люди повинні знати що змінилося і як.

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    Як створити хороший лог змін?

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    Головні принципи

  %ul
    %li
      Лог змін <em>для людей</em>, а не машин.
    %li
      Окремий розділ для кожної версії.
    %li
      Зміни одного типу мають бути згруповані.
    %li
      На версії та секції потрібно ставити гіперпосилання.
    %li
      Остання версія мусить бути на початку.
    %li
      Кожна версія має мати власну дату.
    %li
      Вкажіть чи підтримуєте Ви принципи #{link_to "Семантичне версіювання", data.links.semver}.

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types Типи змін

  %ul
    %li
      %code Додано
      для нового функціоналу.
    %li
      %code Змінено
      для змін в існуючий функціонал.
    %li
      %code Застаріло
      для функціоналу, що планується видалити.
    %li
      %code Видалено
      про вже видалений функціонал.
    %li
      %code Виправлення
      для будь яких виправлень.
    %li
      %code Безпека
      при виявленні вразливостей.

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    Як мені докладати мінімальні зусилля для ведення логу змін?

  %p
    Ведіть розділ <code>Нове</code> на початку файла.

  %p Для переслідування подвійної цілі:

  %ul
    %li
      Люди можуть бачити майбутні зміни в найближчому релізі
    %li
      Як настане час релізу, Ви можете перемістити розділ <code>Нове</code>
      у розділ нового релізу,

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    Чи може лог змін бути поганим?

  %p Так. Ось декілька способів зробити лог змін не вдалим.

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    Логи змін між комітами.

  %p
    Використовувати логи комітів як логи змін - це погана ідея.
    Вони наповнені інформаційним шумом. Такими як коміти злиття,
    не інформативні назви комітів, змінами у документації тощо.

  %p
    Призначення комітів у тому, щоб документувати кроки в еволюції коду.
    У деяких проектах історія комітів доглянута, в інших - ні.

  %p
    Призначення ж логу змін полягає у документації вагомих змін,
    часто між багатьма комітами, доносячи їх призначення до кінцевого
    користувача.

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    Ігнорування застарілого функціоналу

  %p
    Коли люди оновлюються з версії до версії, їм потрібна повна впевненість
    у тому, чи може щось зламатися. Їм повинна бути надана можливість оновитися
    до версії зі списком застарілого функціоналу, видалити все застаріле,
    а потім оновитися до версії з видаленим застарілим функціоналом.

  %p
    Якщо Ви не займаєтеся логом змін, то хоча б ведіть список
    застарілого, видаленого або серйозних змін функціоналу.


  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    Незрозумілі дати

  %p
    Регіональні дати можуть відрізнятися і це може бути складно для
    правильного розуміння ваших дат для користувачів у різних куточках
    світу. Варто надавати переваги датам, що форматовані за таким зразком:
    <code>2017-07-17</code>. У них числа ідуть від найбільшого до
    найменшого: рік, місяць, день. Мінімізує конфузи у випадку використання
    регіональних форматів, коли день і місяць можуть мати різний порядок.
    Цей формат не пересікається з більшістю інших форматів і є
    #{link_to "стандартом ISO", data.links.isodate}. Тому такий формат рекомендується
    для логів змін.

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    Невідповідність змінам

  %p
    Журнал змін, у якому згадуються лише деякі зміни, може становити таку ж небезпеку
    як і відсутність журналу змін. Незважаючи на те, що багато змін можуть бути
    незначними (наприклад, видалення одного пробілу може не потребувати сповіщення),
    будь-які важливі зміни слід згадувати в журналі змін. Застосовуючи зміни
    непослідовно ваші користувачі можуть помилково подумати, що журнал змін
    є єдиним джерелом правди. І так має бути. Сила тут походить від відповідальності
    &mdash; мати хороший журнал змін означає постійно оновлювати журнал змін.

  %aside
    Є також інші. Допоможіть зібрати антипатерни,
    = link_to "створіть тікет", data.links.issue
    або пул реквест

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    Поширені запитання

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    Чи існує стандарт логів змін?

  %p
    Не зовсім. Є   #{link_to "стайлгайд логів змін GNU", data.links.gnustyle}
    або  #{link_to "довгий у два параграфа GNU NEWS file", data.links.gnunews}.
    Обидва неадекватні і неповні.

  %p
    Цей проект покликаний бути
    = link_to "поліпшеною угодою про логи змін.", data.links.changelog
    Це виходить із ліпших практик open source спільноти.

  %p
    Здорова критика, дискусія та пропозиції поліпшення
    = link_to "вітаються.", data.links.issue

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    Як назвати файл логів змін?

  %p
    Назвіть <code>CHANGELOG.md</code>. У деяких проектах файл носить назви
    <code>HISTORY</code>, <code>NEWS</code> або <code>RELEASES</code>.

  %p
    Може видатися, що назва файлу для логів змін не суттєва, проте
    навіщо ускладнювати життя користувачам змушуючи їх шукати?

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    Як щодо GitHub релізів?

  %p
    Це чудова ініціатива. #{link_to "Релізи", data.links.github_releases} можуть бути використані для
    перетворення простих тегів у Git (<code>v1.0.0</code> - до прикладу)
    у деталізовані нотатки до релізів шляхом їх редагування вручну або
    за допомогою коментарів до цих тегів.

  %p
    Релізи GitHub є не портативним логом змін, який може бути показаний
    користувачам лише на самому сайті GitHub. Його можна вести подібно
    до формату Keep a Changelog, але для цього потрібні значні зусилля.

  %p
    Поточна версія на GitHub не так добре очевидна для користувача,
    на відмінну від звичайних файлів з іменами у верхньому регістрі
    (<code>README</code>, <code>CONTRIBUTING</code> і так далі).
    Крім того, інтерфейс не дозволяє посилання на логи комітів між
    релізами.

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    Чи можуть логи змін автоматично парситися?

  %p
    Це заскладно, оскільки люди використовують різні формати та
    імена файлів.

  %p
    #{link_to "Vandamme", data.links.vandamme} - це гем для Ruby, створений
    командою Gemnasium, який парсить багато (але не всі) логи
    змін проектів з відкритим кодом.

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    А як щодо yanked-релізів?

  %p
    Yanked-релізи - це версії, що вилучаються із-за серйозних багів або проблем
    з безпекою у них. Часто вони навіть не згадуються у логах змін.
    А повинні. І описуватися вони повинні так:

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    <code>[YANKED]</code> навмисне кидається в очі. Важливо, щоб його
    помітили. Обмежений квадратними дужками, щоб його легше було
    спарсити.


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    Чи є сенс переписати лог змін?

  %p
    Звісно. Завжди є сенс поліпшувати лог змін. І відкривати пул-реквести
    додаючи втрачені релізи у проекти з відкритим кодом і закинутими
    логами змін.

  %p
    Також можлива ситуація, що Ви забули вказати критичні зміни для версії.
    Очевидно, що потрібно такий лог оновити.



  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    Як я можу допомогти вашому проекту?

  %p
    Цей документ не претендує на виключну <strong>правду</strong>;
    Це моє бачення, зі зібраною інформацію та прикладами.

  %p
    Хотів би, щоб спільнота дійшла згоди. Я вірю у дискусію та результат.

  %p
    Тому <strong>#{link_to "беріть участь", data.links.repo}</strong>.

.press
  %h3 Обговорення
  %p
    Я приходив на #{link_to "підкаст The Changelog", data.links.thechangelog}, щоб обговорити
    те, чому ментейнери та контриб'ютори повинні вести логи змін, а також
    про мою мотивацію для створення цього проекту.
```

## File: source/zh-CN/0.3.0/index.html.haml
```haml
---
title: 如何维护更新日志
description: 更新日志绝对不应该是git日志的堆砌物
language: zh-CN
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### 更新日志是什么？
  更新日志（Change Log）是一个由人工编辑，以时间为倒序的列表。
  这个列表记录所有版本的重大变动。

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### 为何要提供更新日志？
  为了让用户和开发人员更好知道每一个版本有哪些区别。

  ### 为何我要在乎呢？
  因为归根结底软件是为人提供的。既然你不关心人，那么为何写软件呢？
  多少你还是要关心你的受众。

  本文档原作者和另外两个人的一个[播客][thechangelog]向大家介绍了，
  为何代码的管理者和开发者应该在乎更新日志。如果你有一小时时间和很好的英文听力本领，
  不妨听听。

  ### 怎么定义好的更新日志
  好问题！

  一个好的更新日志，一定满足：

  - 给人而不是机器写的。记住，要说人话。
  - 快速跳转到任意段。所以采用markdown格式
  - 一个版本对应一个章节。
  - 最新的版本在上，最老的在下面。
  - 所有日期采用'YYYY-MM-DD'这种规范。（例如北京奥运会的2008年8月8日是2008-08-08）这个是国际通用，任何语言
  都能理解的，并且还被[xkcd](https://xkcd.com/1179/)推荐呢！
  - 标出来是否遵守[语义化版本格式][semver]
  - 每一个软件的版本必须：
    - 标明日期（要用上面说过的规范）
    - 标明分类（采用英文）。规范如下：
      - 'Added' 添加的新功能
      - 'Changed' 功能变更
      - 'Deprecated' 不建议使用，未来会删掉
      - 'Removed' 之前不建议使用的功能，这次真的删掉了
      - 'Fixed' 改的bug
      - 'Security' 改的有关安全相关bug


  ### 怎么尽可能减少耗费的精力？
  永远在文档最上方提供一个'Unreleased' 未发布区域，来记录当前的变化。
  这样作有两大意义。

  - 大家可以看到接下来会有什么变化
  - 在发布时，只要把'Unreleased'改为当前版本号，然后再添加一个新的'Unreleased'就行了


  ### 吐槽环节到了
  请你一定要注意：

  - **把git日志扔到更新日志里。**看似有用，然并卵。
  - **不写'deprecations'就删功能。**不带这样坑队友的。
  - **采用各种不靠谱日期格式** 2012年12月12日，也就中国人能看懂了。

  如果你还有要吐槽的，欢迎留[issue][issues]或者直接PR


  ### 世界上不是有标准的更新日志格式吗？
  貌似GNU或者GNU NEWS还是提过些规范的，事实是它们太过简陋了。
  开发有那么多种情况，采用那样的规范，确实是不太合适的。

  这个项目提供的[规范][CHANGELOG]是作者本人希望能够成为世界规范的。
  作者不认为当前的标准足够好，而且作为一个社区，我们是有能力提供更棒的规范。
  如果你对这个规范有不满的地方，不要忘记还可以[吐槽][issues]呢。

  ### 更新日志文件名应该叫什么？

  我们的案例中给的名字就是最好的规范：`CHANGELOG.md`，注意大小写。

  像`HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md`这么
  多文件名就太不统一了。

  只会让大家更难找到。

  ### 为何不直接记录`git log`?

  因为git日志一定是乱糟糟的。哪怕一个最理想的由完美的程序员开发和提交的，哪怕一个
  从不忘记每次提交全部文件，不写错别字，不忘记重构每一个部分，也无法保证git日志完美无瑕。
  况且git日志的核心在于记录代码的演化，而更新日志则是记录最重要的变更。

  就像注释之于代码，更新日志之于git日志。前者解释*为什么*，而后者说明*发生了什么*。


  ### 更新日志能机器识别吗？
  非常困难，因为有各种不同的文件格式和其它规范。

  [Vandamme][vandamme]是一个Ruby程序（由[Gemnasium][gemnasium]团队制作），可以解析
  好多种（但绝对不是全部）开源库的更新日志。

  ### 到底是CHANGELOG还是更新日志

  CHANGELOG是文件名，更新日志是常用说法。CHANGELOG采用大写是有历史根源的。就像很多类似的文件
  [`README`][README]， [`LICENSE`][LICENSE]，还有[`CONTRIBUTING`][CONTRIBUTING]。

  采用大写可以更加显著，毕竟这是项目很重要的元信息。就像[开源徽章][shields]。

  ### 那些后来撤下的版本怎么办？
  因为各种安全/重大bug原因被撤下的版本被标记'YANKED'。这些版本一般不出现在更新日志里，但作者建议他们出现。
  显示方式应该是：

  `## [0.0.5] - 2014-12-13 [YANKED]`

  `[YANKED]`采用大写更加显著，因为这个信息很重要。而采用方括号则容易被程序解析。

  ### 是否可以重写更新日志
  当然。哪怕已经上线了，也可以重新更新更新日志。有许多开源项目更新日志不够新，所以作者就会帮忙更新。

  另外，很有可能你忘记记录一个重大功能更新。所以这时候应该去重写更新日志。

  ### 如何贡献？
  本文档并不是**真理**。这只是原作者的个人建议，并且包括许多收集的例子。
  哪怕[本开源库][gh]提供一个[更新日志案例][CHANGELOG]，我刻意没有提供一个
  过于苛刻的规则列表（不像[语义化版本格式][semver]）。

  这是因为我希望通过社区达到统一观点，我认为中间讨论的过程与结果一样重要。

  所以[欢迎贡献][gh]。


  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/lang/zh-CN/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme/
```

## File: source/zh-CN/1.0.0/index.html.haml
```haml
---
title: 如何维护更新日志
description: 更新日志绝对不应该是 git 日志的堆砌物
language: zh-CN
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    更新日志是什么？

  %p
    更新日志（Change Log）是一个由人工编辑、以时间为倒序的列表，用于记录项目中每个版本的显著变动。

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    为何要提供更新日志？

  %p
    为了让用户和开发人员更简单清晰地知晓项目的不同版本之间有哪些显著变动。

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    哪些人需要更新日志？

  %p
    人人需要更新日志。无论是用户还是开发者。当软件有变动时，大家希望知道改动是为何、以及如何进行的。

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    怎样制作高质量的更新日志？

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    指导原则

  %ul
    %li
      记住日志是写给<em>人</em>而非机器的。
    %li
      每个版本都应该有独立的入口。
    %li
      同类改动应该分组放置。
    %li
      不同版本应分别设置链接。
    %li
      新版本在前，旧版本在后。
    %li
      应包括每个版本的发布日期。
    %li
      注明是否遵守#{link_to "语义化版本规范", data.links.semver}。

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 变动类型

  %ul
    %li
      %code Added
      新添加的功能。
    %li
      %code Changed
      对现有功能的变更。
    %li
      %code Deprecated
      已经不建议使用，即将移除的功能。
    %li
      %code Removed
      已经移除的功能。
    %li
      %code Fixed
      对 bug 的修复。
    %li
      %code Security
      对安全性的改进。

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    如何减少维护更新日志的精力？

  %p
    在文档最上方提供 <code>Unreleased</code> 区块以记录即将发布的更新内容。

  %p 这样做有两个好处：

  %ul
    %li
      大家可以知道在未来版本中可能会有哪些变更。
    %li
      在发布新版本时，直接将 <code>Unreleased</code> 区块中的内容移动至新发布版本的描述区块就可以了。

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    有很糟糕的更新日志吗？

  %p 当然有，下面就是一些糟糕的方式。

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    使用 git 日志

  %p
    使用 git 日志作为更新日志是个非常糟糕的方式：git 日志充满各种无意义的信息，如合并提交、语焉不详的提交标题、文档更新等。

  %p
    提交的目的是记录源码的演化。一些项目会清理提交记录，一些则不会。

  %p
    更新日志的目的则是记录重要的变更以供受众阅读，记录范围通常涵盖多次提交。

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    无视即将弃用的功能

  %p
    当从一个版本升级至另一个时，人们应清楚（尽管痛苦）地知道哪些部分将不再被支持。应该允许先升级至一个列出哪些功能将会被弃用的版本，待去掉那些不再支持的部分后，再升级至把那些弃用功能真正移除的版本。

  %p
    即使其他什么都不做，也至少要在更新日志中列出 deprecations，removals 以及其他重大变动。

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    易混淆的日期格式

  %p
    不同区域有着不同的时间格式，要找到让大家都满意的日期格式不是件容易的事。<code>2012-06-02</code>
    的格式从大到小排列符合逻辑、不容易与其他日期格式混淆，而且还符合
    #{link_to "ISO 标准", data.links.isodate}。因此，推荐在更新日志中采用使用此种日期格式。

  %aside
    还有更多内容？请通过
    = link_to "Issues", data.links.issue
    或是 Pull Request 协助收集。

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    常见问题

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    是否有一个标准化的更新日志格式？

  %p
    并没有。虽然 GNU 提供了更新日志样式指引，以及那个仅有两段长的 GNU NEWS 文件“指南”，但两者均远远不够。

  %p
    此项目旨在提供一个
    #{link_to "更好的更新日志范例", data.links.changelog}，所有点子都来自于对开源社区中优秀实例的观察与记录。

  %p
    欢迎#{link_to "提供", data.links.issue}有建设性的批评、讨论及建议。

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    更新日志文件应被如何命名？

  %p
    通常使用 <code>CHANGELOG.md</code>。有些项目将其命名为
    <code>HISTORY</code>、<code>NEWS</code> 或是 <code>RELEASES</code>。

  %p
    当然，你可能认为更新日志的命名并不那么重要，但为什么要为难那些仅仅是想看到都有哪些重大变更的用户呢？

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    GitHub Releases 怎么样？

  %p
    这是个非常好的提议。#{link_to "GitHub Releases", data.links.github_releases}
    可通过手动添加发布日志或将带有注释的 git 标签信息抓取后转换的方式，将简单的
    git 标签（如一个叫 <code>v1.0.0</code> 的标签）转换为信息丰富的发布日志。

  %p
    GitHub Releases 会创建一个非便携、仅可在 GitHub 环境下显示的更新日志。尽管会花费更多时间，但将之处理成更新日志格式是完全可能的。

  %p
    现行版本的 GitHub Releases 不像那些典型的大写文件（<code>README</code>，<code>CONTRIBUTING</code>
    等），仍可以认为是不利于用户探索的。另一个小问题则是目前的 GitHub Releases 界面并没有提供不同版本间的 commit 日志链接。

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    更新日志可以被自动识别吗？

  %p
    非常困难，因为有各种不同的文件格式和命名。


  %p
    #{link_to "Vandamme", data.links.vandamme} 是一个 Ruby 程序，由
    Gemnasium 团队制作，可以解析多种（但绝对不是全部）开源库的更新日志。


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    那些后来撤下的版本怎么办？

  %p
    因重大 bug 或安全性原因而被撤下的版本通常不会出现在更新日志中，但仍然建议记录下来。你可以这样作出记录：

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    因为这类更改十分重要，所以 <code>[YANKED]</code> 标签应该非常醒目。此外，用方括号包围可使其更易被程序识别。


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    是否可以重写更新日志？

  %p
    当然可以。总会有合适的原因去改进更新日志。我也时常提
    Pull Request 来为那些未维护更新日志的开源项目加入缺失的发布信息。

  %p
    另外，你很有可能发现自己忘记记录一个重大功能更新。这种情况下显然应该重写更新日志。


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    如何贡献？

  %p
    本文档并非<strong>真理</strong>。而是我深思熟虑后的建议，以及我收集的信息与样例。

  %p
    希望我们的社区可以对此达成一致。我相信讨论的过程与最终结果一样重要。

  %p
    所以欢迎<strong>#{link_to "贡献", data.links.repo}</strong>。

.press
  %h3 访谈
  %p
    我在 #{link_to "The Changelog podcast", data.links.thechangelog}
    上讲述了为何维护者与贡献者应关心更新日志，以及这个项目背后的动机。
```

## File: source/zh-CN/1.1.0/index.html.haml
```haml
---
title: 如何维护更新日志
description: 更新日志绝对不应该是 git 日志的堆砌物
language: zh-CN
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    更新日志是什么？

  %p
    更新日志（Change Log）是一个由人工编辑、以时间为倒序的列表，用于记录项目中每个版本的显著变动。

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    为何要提供更新日志？

  %p
    为了让用户和开发人员更简单清晰地知晓项目的不同版本之间有哪些显著变动。

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    哪些人需要更新日志？

  %p
    人人需要更新日志。无论是用户还是开发者。当软件有变动时，大家希望知道改动是为何、以及如何进行的。

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    怎样制作高质量的更新日志？

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    指导原则

  %ul
    %li
      记住日志是写给<em>人</em>而非机器的。
    %li
      每个版本都应该有独立的入口。
    %li
      同类改动应该分组放置。
    %li
      不同版本应分别设置链接。
    %li
      新版本在前，旧版本在后。
    %li
      应包括每个版本的发布日期。
    %li
      注明是否遵守#{link_to "语义化版本规范", data.links.semver}。

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 变动类型

  %ul
    %li
      %code Added
      新添加的功能。
    %li
      %code Changed
      对现有功能的变更。
    %li
      %code Deprecated
      已经不建议使用，即将移除的功能。
    %li
      %code Removed
      已经移除的功能。
    %li
      %code Fixed
      对 bug 的修复。
    %li
      %code Security
      对安全性的改进。

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    如何减少维护更新日志的精力？

  %p
    在文档最上方提供 <code>Unreleased</code> 区块以记录即将发布的更新内容。

  %p 这样做有两个好处：

  %ul
    %li
      大家可以知道在未来版本中可能会有哪些变更。
    %li
      在发布新版本时，直接将 <code>Unreleased</code> 区块中的内容移动至新发布版本的描述区块就可以了。

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    有很糟糕的更新日志吗？

  %p 当然有，下面就是一些糟糕的方式。

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    使用 git 日志

  %p
    使用 git 日志作为更新日志是个非常糟糕的方式：git 日志充满各种无意义的信息，如合并提交、语焉不详的提交标题、文档更新等。

  %p
    提交的目的是记录源码的演化。一些项目会清理提交记录，一些则不会。

  %p
    更新日志的目的则是记录重要的变更以供受众阅读，记录范围通常涵盖多次提交。

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    无视即将弃用的功能

  %p
    当从一个版本升级至另一个时，人们应清楚（尽管痛苦）地知道哪些部分将不再被支持。应该允许先升级至一个列出哪些功能将会被弃用的版本，待去掉那些不再支持的部分后，再升级至把那些弃用功能真正移除的版本。

  %p
    即使其他什么都不做，也至少要在更新日志中列出 deprecations，removals 以及其他重大变动。

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    易混淆的日期格式

  %p
    不同区域有着不同的时间格式，要找到让大家都满意的日期格式不是件容易的事。<code>2012-06-02</code>
    的格式从大到小排列符合逻辑、不容易与其他日期格式混淆，而且还符合
    #{link_to "ISO 标准", data.links.isodate}。因此，推荐在更新日志中采用使用此种日期格式。

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    不一致的变更

  %p
    仅提到部分变更的更新日志可能和没有更新日志一样危险。
    虽然许多变更可能并不相关——例如，删除一个空格可能在所有情况下都不需要被记录下来——但任何重要的变更都应该在更新日志中提及。
    通过不一致地应用变更，你的用户可能会错误地认为更新日志是事实的唯一来源。
    理应如此。能力越大，责任越大——拥有一个好的更新日志意味着拥有一个一致性更新的更新日志。

  %aside
    还有更多内容？请通过
    = link_to "Issues", data.links.issue
    或是 Pull Request 协助收集。

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    常见问题

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    是否有一个标准化的更新日志格式？

  %p
    并没有。虽然有#{link_to "GNU 更新日志指南", data.links.gnustyle}，以及那个仅有两段长的 #{link_to "GNU - The NEWS File 指南", data.links.gnunews}，但两者均远远不够。

  %p
    此项目旨在提供一个
    #{link_to "更好的更新日志范例", data.links.changelog}，所有点子都来自于对开源社区中优秀实例的观察与记录。

  %p
    欢迎#{link_to "提供", data.links.issue}有建设性的批评、讨论及建议。

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    更新日志文件应被如何命名？

  %p
    通常使用 <code>CHANGELOG.md</code>。有些项目将其命名为
    <code>HISTORY</code>、<code>NEWS</code> 或是 <code>RELEASES</code>。

  %p
    当然，你可能认为更新日志的命名并不那么重要，但为什么要为难那些仅仅是想看到都有哪些重大变更的用户呢？

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    GitHub Releases 怎么样？

  %p
    这是个非常好的提议。#{link_to "GitHub Releases", data.links.github_releases}
    可通过手动添加发布日志或将带有注释的 git 标签信息抓取后转换的方式，将简单的
    git 标签（如一个叫 <code>v1.0.0</code> 的标签）转换为信息丰富的发布日志。

  %p
    GitHub Releases 会创建一个非便携、仅可在 GitHub 环境下显示的更新日志。尽管会花费更多时间，但将之处理成更新日志格式是完全可能的。

  %p
    现行版本的 GitHub Releases 不像那些典型的大写文件（<code>README</code>，<code>CONTRIBUTING</code>
    等），仍可以认为是不利于用户探索的。另一个小问题则是目前的 GitHub Releases 界面并没有提供不同版本间的 commit 日志链接。

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    更新日志可以被自动识别吗？

  %p
    非常困难，因为有各种不同的文件格式和命名。


  %p
    #{link_to "Vandamme", data.links.vandamme} 是一个 Ruby 程序，由
    Gemnasium 团队制作，可以解析多种（但绝对不是全部）开源库的更新日志。


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    那些后来撤下的版本怎么办？

  %p
    因重大 bug 或安全性原因而被撤下的版本通常不会出现在更新日志中，但仍然建议记录下来。你可以这样作出记录：

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    因为这类更改十分重要，所以 <code>[YANKED]</code> 标签应该非常醒目。此外，用方括号包围可使其更易被程序识别。


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    是否可以重写更新日志？

  %p
    当然可以。总会有合适的原因去改进更新日志。我也时常提
    Pull Request 来为那些未维护更新日志的开源项目加入缺失的发布信息。

  %p
    另外，你很有可能发现自己忘记记录一个重大功能更新。这种情况下显然应该重写更新日志。


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    如何贡献？

  %p
    本文档并非<strong>真理</strong>。而是我深思熟虑后的建议，以及我收集的信息与样例。

  %p
    希望我们的社区可以对此达成一致。我相信讨论的过程与最终结果一样重要。

  %p
    所以欢迎<strong>#{link_to "贡献", data.links.repo}</strong>。

.press
  %h3 访谈
  %p
    我在 #{link_to "The Changelog podcast", data.links.thechangelog}
    上讲述了为何维护者与贡献者应关心更新日志，以及这个项目背后的动机。
```

## File: source/zh-TW/0.3.0/index.html.haml
```haml
---
title: 如何維護更新日誌
description: 更新日誌絕對不應該是 git 日誌的堆砌物
language: zh-TW
version: 0.3.0
---

%h1= current_page.data.title
%h2= current_page.data.description

Version <strong>#{current_page.metadata[:page][:version]}</strong>

:markdown
  ### 更新日誌是什麽？
  更新日誌（Change Log）是一個由人工編輯，以時間為倒敘的列表。
  這個列表記錄所有版本的重大變動。

<pre class="changelog">#{File.read("CHANGELOG.md")}</pre>

:markdown
  ### 為何要提供更新日誌？
  為了讓用戶和開發人員更好知道每一個版本有哪些區別。

  ### 為何我要在乎呢？
  因為歸根結底軟體是為人提供的。既然你不關心人，那麽為何寫軟體呢？
  多少你還是要關心你的受眾。

  本文檔原作者和另外兩個人的一個[部落格][thechangelog]向大家介紹了，
  為何程式碼的管理者和開發者應該在乎更新日誌。如果你有一小時時間和很好的英文聽力本領，
  不妨聽聽。

  ### 怎麽定義好的更新日誌
  好問題！

  一個好的更新日誌，一定滿足：

  - 給人而不是機器寫的。記住，要說人話。
  - 快速跳轉到任意段。所以採用 markdown 格式
  - 一個版本對應一個章節。
  - 最新的版本在上面，最舊的在下面。
  - 所有日期採用 'YYYY-MM-DD' 這種規範。（例如北京奧運會的 2008 年 8 月 8 日是 2008-08-08）這個是國際通用，任何語言
  都能理解的，並且還被 [xkcd](https://xkcd.com/1179/) 推薦呢！
  - 標出來是否遵守[語義化版本格式][semver]
  - 每一個軟體的版本必須：
    - 標明日期（要用上面說過的規範）
    - 標明分類（採用英文）。規範如下：
      - 'Added' 添加的新功能
      - 'Changed' 功能變更
      - 'Deprecated' 不建議使用，未來會刪掉
      - 'Removed' 之前不建議使用的功能，這次真的刪掉了
      - 'Fixed' 修正的 bug
      - 'Security' 修正了安全相關的 bug


  ### 怎麽盡可能減少耗費的精力？
  永遠在文檔最上方提供一個 'Unreleased' 未發布區域，來記錄當前的變化。
  這樣做有兩大意義。

  - 大家可以看到接下來會有什麽變化
  - 在發布時，只要把 'Unreleased' 改為當前版本號，然後再添加一個新的 'Unreleased' 就行了


  ### 吐槽環節到了
  請你一定要注意：

  - **把 git 日誌扔到更新日誌裏。**看似有用，然而並沒有什麼作用。
  - **不寫 'deprecations' 就刪功能。**不該這樣坑隊友的。
  - **採用各種不可靠的日期格式** 2012 年 12 月 12 日，也就懂中文的人能看得懂了。

  如果你還有要吐槽的，歡迎留 [issue][issues] 或者直接 PR


  ### 世界上不是有標準的更新日誌格式嗎？
  貌似 GNU 或者 GNU NEWS 還是提過些規範的，事實是它們太過簡陋了。
  開發有那麽多中情況，採用那樣的規範，確實是不太合適的。

  這個項目提供的[規範][CHANGELOG]是作者本人希望能夠成為世界規範的。
  作者不認為當前的標準足夠好，而且作為一個社區，我們是有能力提供更棒的規範。
  如果你對這個規範有不滿的地方，不要忘記還可以[吐槽][issues]呢。

  ### 更新日誌文件名應該叫什麽？

  我們的案例中給的名字就是最好的規範：`CHANGELOG.md`，注意大小寫。

  像 `HISTORY.txt`, `HISTORY.md`, `History.md`, `NEWS.txt`,
  `NEWS.md`, `News.txt`, `RELEASES.txt`, `RELEASE.md`, `releases.md` 這麽
  多文件名就太不統一了。

  只會讓大家更難找到。

  ### 為何不直接記錄 `git log`?

  因為 git 日誌一定是亂糟糟的。哪怕一個最理想的由完美的程式設計師開發的提交的，哪怕一個
  從不忘記每次提交全部文件，不寫錯別字，不忘記重構每一個部分——也無法保證 git 日誌完美無瑕。
  況且 git 日誌的核心在於記錄程式碼的演化，而更新日誌則是記錄最重要的變更。

  就像註解之於程式碼，更新日誌之於 git 日誌。前者解釋*為什麽*，而後者說明*發生了什麽*。


  ### 更新日誌能機器識別嗎？
  非常困難，因為有各種不同的文件格式和其他規範。

  [Vandamme][vandamme] 是一支 Ruby 程式（由 [Gemnasium][gemnasium] 團隊制作），可以解析
  很多種（但絕對不是全部）開源程式庫的更新日誌。

  ### 到底是 CHANGELOG 還是更新日誌

  CHANGELOG 是文件名，更新日誌是常用說法。CHANGELOG 採用大寫是有歷史根源的。就像很多類似的文件
  [`README`][README]，[`LICENSE`][LICENSE]，還有 [`CONTRIBUTING`][CONTRIBUTING]。

  採用大寫可以更加顯著，畢竟這是項目很重要的 metadata。就像[開源徽章][shields]。

  ### 那些後來撤下的版本怎麽辦？
  因為各種安全/重大 bug 原因被撤下的版本被標記 'YANKED'。這些版本一般不出現在更新日誌裏，但作者建議他們出現。
  顯示方式應該是：

  `## [0.0.5] - 2014-12-13 [YANKED]`

  `[YANKED]` 採用大寫更加顯著，因為這個訊息很重要。而採用方括號則容易被程式解析。

  ### 是否可以重寫更新日誌
  當然。哪怕已經上線了，也可以重新更新更新日誌。有許多開源項目更新日誌不夠新，所以作者就會幫忙更新。

  另外，很有可能你忘記記錄一個重大功能更新。所以這時候應該去重寫更新日誌。

  ### 如何貢獻？
  本文檔並不是**真理**。這只是原作者的個人建議，並且包括許多收集的例子。
  哪怕[本開源庫][gh]提供一個[更新日誌案例][CHANGELOG]，我刻意沒有提供一個
  過於苛刻的規則列表（不像[語義化版本格式][semver]）。

  這是因為我希望通過社區達到統一觀點，我認為中間討論的過程與結果一樣重要。

  所以[歡迎貢獻][gh]。


  [CHANGELOG]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CHANGELOG.md
  [CONTRIBUTING]: https://github.com/olivierlacan/keep-a-changelog/blob/main/CONTRIBUTING.md
  [LICENSE]: https://github.com/olivierlacan/keep-a-changelog/blob/main/LICENSE
  [README]: https://github.com/olivierlacan/keep-a-changelog/blob/main/README.md
  [gemnasium]: https://gemnasium.com/
  [gh]: https://github.com/olivierlacan/keep-a-changelog
  [issues]: https://github.com/olivierlacan/keep-a-changelog/issues
  [semver]: https://semver.org/lang/zh-CN/
  [shields]: https://shields.io/
  [thechangelog]: https://changelog.com/podcast/127
  [vandamme]: https://github.com/tech-angels/vandamme
```

## File: source/zh-TW/1.0.0/index.html.haml
```haml
---
title: 如何維護更新日誌
description: 別讓你的更新日誌成為 git log 的雙胞胎
language: zh-TW
version: 1.0.0
---
.header
  .title
    %h1= current_page.data.title
    %h2= current_page.data.description

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    更新日誌是什麼？

  %p
    更新日誌（Changelog）是說明專案在開發過程中，版本之間的差異紀錄，由開發人員由新而舊撰寫。

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    為什麼需要提供更新日誌？

  %p
    為了讓使用者和開發人員更簡單明確地了解各個版本之間有著哪些改動。

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    哪些人需要更新日誌？

  %p
    大家都需要更新日誌。

  %p
    無論是開發人員或是使用者，都會在意軟體包含了什麼東西。
    當軟體更新了，大家都想知道改了些什麼以及為什麼要改。

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    如何寫出高品質的日誌？

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    撰寫原則

  %ul
    %li
      更新日誌是寫給「人」看的，不是機器。
    %li
      每個版本都應該有獨立的進入點。
    %li
      相同類型的改動應該被分類在同一組。
    %li
      版本與章節應「可連結化」。
    %li
      新版本總是寫在最前面。
    %li
      每個版本都應該註記發佈日期。
    %li
      版本命名應遵守#{link_to "語意化版本", data.links.semver}格式。

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 改動類型

  %ul
    %li
      %code Added
      ：當增加了新功能。
    %li
      %code Changed
      ：當更動了既有的功能。
    %li
      %code Deprecated
      ：當功能將在近期被移除。
    %li
      %code Removed
      ：當移除了現有的功能。
    %li
      %code Fixed
      ：當修復了某些錯誤。
    %li
      %code Security
      ：當增進了安全性漏洞。

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    如何提升維護「更新日誌」的效率？

  %p
    在日誌上方使用 <code>Unreleased</code> 區塊記錄即將發佈的更新內容。

  %p 這麼做能夠：

  %ul
    %li
      讓大家知道在未來的版本中可能會有哪些改動。
    %li
      發佈新版本時，直接將 <code>Unreleased</code> 移到新版本的區塊就完成了 ヾ(*´ω｀*)ノ

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    不就只是個日記而已嗎？難道可以把日誌得很糟嗎？

  %p 當然，下面有些糟糕的範例：

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    🚫 直接複製 git log 到更新日誌

  %p
    直接把 git log 複製一份成更新日誌絕對不是個好主意。

  %p  
    git log 充滿了各種瑣碎的訊息，像 merge commits、亂七八糟的提交訊息或是文件更新等。

  %p
    Commits 的目的應該是記錄原始碼演進的過程，有些專案會清理 commits，但有些不會。

  %p
    更新日誌的目的則是記錄那些值得一提的改動，經常涵蓋多個 commits，最終目的仍然是讓看的人一目了然。

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    🚫 忽略 Deprecations

  %p
    當使用者升級版本時，他應該要能預先知道哪些環節可能會出問題，然而，在理想的狀況下，應該讓使用者有空間能預先升級即將被棄用的功能，替換掉棄用功能之後，再升級至棄用功能被真正移除的版本。

  %p
    即使不這麼做，也要在更新日誌中列出棄用的、移除的或是任何可能導致程式碼失效的重大改動。

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    🚫 容易混淆的日期格式

  %p
    在世界的每個角落，不同區域有著不同的時間格式，找到讓大家都滿意的日期格式不是件簡單的事。使用像
    <code>2017-07-17</code> 的格式能清楚傳達日期，而且不易與其他日期格式混淆，同時也遵守
    #{link_to "ISO 標準", data.links.isodate}，因此推薦使用像這樣的日期格式。

  %aside
    其實還有許多應該避免的。大家可以透過
    = link_to "Issue", data.links.issue
    或是 Pull Request 協助蒐集 ฅ' ω 'ฅ

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    常見問題

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    有沒有標準格式可以參考呢？

  %p
    並沒有。雖然有 #{link_to "GNU 更新日誌指南", data.links.gnu} 以及只有兩段長的
    #{link_to "GNU - The NEWS File 指南", data.links.gnunews}（括弧笑），但這些並不足以稱為「標準」。

  %p
    這項專案的宗旨在於提供一個
    #{link_to "更好的更新日誌範例", data.links.changelog}，源於觀察開源社群中優秀的實際案例，把它們蒐集在一起。

  %p
    歡迎各位#{link_to "提供", data.links.issue}有建設性的建議和批評。

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    更新日誌的檔案名稱應該是？

  %p
    通常使用 <code>CHANGELOG.md</code>。也有用
    <code>HISTORY</code>、<code>NEWS</code>、或是 <code>RELEASES</code> 的例子。

  %p
    或許你認為取什麼名字並不是件多麼重要的事，但為什麼要讓只是想看日誌的使用者不容易找到它呢？

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    那麼 GitHub Releases 呢？

  %p
    這是個好問題。#{link_to "GitHub Releases", data.links.github_releases}
    能手動在簡單的 git tag（如
    <code>v1.0.0</code>） 上附加豐富的版本資訊，也能把附帶的 tag
    messages 轉換成漂亮的日誌格式。

  %p
    GitHub Releases 產生的日誌只能在 GitHub 上瀏覽，雖然 GitHub Releases
    能做出接近本專案範例的日誌格式，但這會增加些許與 GitHub 的相依性。

  %p
    現行的 GitHub Releases
    畢竟不像典型的大寫文件（<code>README</code>、<code>CONTRIBUTING</code>
    之類的），按理說會增加使用者找到的難度。另外還有個小問題，目前 GitHub
    Releases 頁面上並沒有提供兩版版本之間 commit logs 的連結。

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    更新日誌能被自動生成嗎？

  %p
    非常困難，各式各樣的提交訊息和檔案名稱難以完全掌握。

  %p
    另外，有些開源專案使用由 Gemnasium
    團隊開發的 #{link_to "Vandamme", data.links.vandamme}
    轉換更新日誌，或許可以當作參考。

  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    那麼被撤下的版本呢？

  %p
    因為重大漏洞或安全性問題而被撤下（unpublished）的版本通常不會出現在日誌裡，但建議仍然記錄下來。你可以這樣記錄它們：

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    其中 <code>[YANKED]</code> 標記應該和原因顯眼地標示在一起，讓使用者注意到它是最重要的事。此外，用中括弧能讓轉換用的程式更容易辨認它們。

  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    可以更改過去版本的日誌內容嗎？

  %p
    當然可以，總是會有好的原因來改善以往寫下的日誌。我也時常發 pull request
    給更新日誌不齊全的開源專案。

  %p
    偶爾會發現自己遺漏了某項重大更新的紀錄，很明顯你應該補齊它們。

  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    我能做些什麼嗎？

  %p
    這份文件並不是《真理》，而是我經過深思熟慮、遵循蒐集到的資訊和範例之後提出的建議。

  %p
    源於我期望社群能達到共識，我相信討論的過程與結果一樣重要。

  %p
    所以，<strong>#{link_to "加入我們", data.links.repo}</strong>吧 ٩(｡・ω・｡)﻿و

.press
  %h3 訪談
  %p
    我在 #{link_to "The Changelog podcast", data.links.thechangelog} 上講述了為什麼維護者與協作者應該在意更新日誌，以及建立這項專案背後的契機。
```

## File: source/zh-TW/1.1.0/index.html.haml
```haml
---
title: 如何維護更新日誌
description: 更新日誌絕對不應該是 git log 的堆砌物
language: zh-TW
version: 1.1.0
---
.header
  .title
    %h1= current_page.data.title
    %h2 更新日誌絕對不應該是 git log 的堆砌物

  = link_to data.links.changelog do
    Version
    %strong= current_page.metadata[:page][:version]

  %pre.changelog{ lang: "en" }= File.read("CHANGELOG.md")

.answers
  %h3#what
    %a.anchor{ href: "#what", aria_hidden: "true" }
    更新日誌是什麼？

  %p
    更新日誌（Change Log）是一個由人工編輯、以時間為倒序的列表，用於記錄專案中每個版本的顯著改動。

  %h3#why
    %a.anchor{ href: "#why", aria_hidden: "true" }
    為什麼需要提供更新日誌？

  %p
    為了讓使用者和開發人員更簡單明確地了解各個版本之間有著哪些顯著改動。

  %h3#who
    %a.anchor{ href: "#who", aria_hidden: "true" }
    哪些人需要更新日誌？

  %p
    大家都需要更新日誌。無論是開發人員或是使用者，都會在意軟體包含了什麼東西。當軟件有變動時，大家想知道改了些什麼以及為什麼要改。

.good-practices
  %h3#how
    %a.anchor{ href: "#how", aria_hidden: "true" }
    怎樣寫出高品質的更新日誌？

  %h4#principles
    %a.anchor{ href: "#principles", aria_hidden: "true" }
    撰寫原則

  %ul
    %li
      記住日誌是寫給<em>人</em>而非機器的。
    %li
      每個版本都應該有獨立的入口。
    %li
      同類改動應該分組放置。
    %li
      版本與章節應「可連結化」。
    %li
      新版本在前，舊版本在後。
    %li
      每個版本都應該註記發佈日期。
    %li
      版本命名應遵守#{link_to "語義化版本規範", data.links.semver}。

  %a.anchor{ href: "#types", aria_hidden: "true" }
  %h4#types 改動類型

  %ul
    %li
      %code Added
      當增加了新功能。
    %li
      %code Changed
      當更動了既有的功能。
    %li
      %code Deprecated
      當功能將在近期被移除。
    %li
      %code Removed
      當移除了現有的功能。
    %li
      %code Fixed
      當修復了某些錯誤。
    %li
      %code Security
      當增進了安全性漏洞。

.effort

  %h3#effort
    %a.anchor{ href: "#effort", aria_hidden: "true" }
    如何提升維護「更新日誌」的效率？

  %p
    在日誌最上方提供 <code>Unreleased</code> 區塊以記錄即將發佈的更新內容。

  %p 這樣做有兩個好處：

  %ul
    %li
      讓大家知道在未來的版本中可能會有哪些改動。
    %li
      在發佈新版本時，直接將 <code>Unreleased</code> 區塊中的內容移動至新發佈版本的描述區塊就可以了。

.bad-practices
  %h3#bad-practices
    %a.anchor{ href: "#bad-practices", aria_hidden: "true" }
    有很糟糕的更新日誌嗎？

  %p 當然有，下面就是一些糟糕的範例。

  %h4#log-diffs
    %a.anchor{ href: "#log-diffs", aria_hidden: "true" }
    🚫 直接複製 git log 到更新日誌

  %p
    使用 git 日誌作為更新日誌是個非常糟糕的方式：git 日誌充滿各種無意義的信息，如合並提交、語焉不詳的提交標題、文檔更新等。

  %p
    提交的目的是記錄源碼的演化。一些專案會清理提交記錄，但有些不會。

  %p
    更新日誌的目的則是記錄重要的改動以供受衆閱讀，通常涵蓋多個提交記錄，最終目的仍然是讓看的人一目了然。

  %h4#ignoring-deprecations
    %a.anchor{ href: "#ignoring-deprecations", aria_hidden: "true" }
    🚫 忽略即將棄用的功能

  %p
    當從一個版本升級至另一個時，使用者應清楚（盡管痛苦）地知道哪些部分將不再被支援。應該允許先升級至一個列出哪些功能將會被棄用的版本，待去掉那些不再支持的部分後，再升級至把那些棄用功能真正移除的版本。

  %p
    即使其他什麼都不做，也至少要在更新日誌中列出棄用的、移除的或是任何可能導致程式碼失效的重大改動。

  %h4#confusing-dates
    %a.anchor{ href: "#confusing-dates", aria_hidden: "true" }
    🚫 容易混淆的日期格式

  %p
    不同區域有著不同的時間格式，要找到讓大家都滿意的日期格式不是件容易的事。使用像<code>2012-06-02</code>
    的格式能清楚傳達日期，而且不易與其他日期格式混淆，同時也遵守
    #{link_to "ISO 標準", data.links.isodate}。因此，推薦在更新日誌中使用此種日期格式。

  %h4#inconsistent-changes
    %a.anchor{ href: "#inconsistent-changes", aria_hidden: "true" }
    🚫 不一致的改動

  %p
    僅提到部分改動的更新日誌可能和沒有更新日誌一樣危險。
    雖然許多改動也許與更新日誌無關——例如，刪除一個空格可能不需要每次都被記錄下來——但任何重要的改動都應該在更新日誌中被提及。
    通過不一致地實施改動，使用者可能會錯誤地認為更新日誌是事實的唯一來源（而它也應該是）。
    能力越大，責任越大——擁有一個好的更新日誌意味著擁有一個一致性更新的更新日誌。

  %aside
    還有更多內容？請通過
    = link_to "opening an issue", data.links.issue
    或是 Pull Request 協助收集。

.frequently-asked-questions
  %h3#frequently-asked-questions
    %a.anchor{ href: "#frequently-asked-questions", aria_hidden: "true" }
    常見問題

  %h4#standard
    %a.anchor{ href: "#standard", aria_hidden: "true" }
    是否有一個標準化的更新日誌格式？

  %p
    並沒有。雖然有 #{link_to "GNU 更新日誌指南", data.links.gnustyle} 以及只有兩段長的
    #{link_to "GNU - The NEWS File 指南", data.links.gnunews}，但這些並不足以稱為「標準」。

  %p
    此專案旨在提供一個
    #{link_to "更好的更新日誌範例", data.links.changelog}，所有點子都來自於對開源社區中優秀實例的觀察與記錄。

  %p
    歡迎各位#{link_to "提供", data.links.issue}有建設性的批評、討論及建議。

  %h4#filename
    %a.anchor{ href: "#filename", aria_hidden: "true" }
    更新日誌的檔案名稱應該是？

  %p
    通常使用 <code>CHANGELOG.md</code>。有些專案將其命名為
    <code>HISTORY</code>、<code>NEWS</code> 或是 <code>RELEASES</code>。

  %p
    當然，你可能認為更新日誌的命名並不那麼重要，但為什麼要為難那些僅僅是想看到都有哪些重大改動的使用者呢？

  %h4#github-releases
    %a.anchor{ href: "#github-releases", aria_hidden: "true" }
    GitHub Releases 怎麼樣？

  %p
    這是個非常好的提議。#{link_to "GitHub Releases", data.links.github_releases}
    可通過手動添加發佈日誌或將帶有註釋的 git 標簽信息抓取後轉換的方式，將簡單的
    git 標簽（如一個叫 <code>v1.0.0</code> 的標簽）轉換為信息豐富的發佈日誌。

  %p
    GitHub Releases 產生的日誌只能在 GitHub 上瀏覽，雖然 GitHub Releases
    能做出接近本專案範例的日誌格式，但這會增加些許與 GitHub 的相依性。

  %p
    現行版本的 GitHub Releases 不像那些典型的大寫文件（<code>README</code>，<code>CONTRIBUTING</code>
    等），仍可以認為是不利於使用者探索的。另一個小問題則是目前的 GitHub Releases 界面並沒有提供不同版本間的 commit 日誌的連結。

  %h4#automatic
    %a.anchor{ href: "#automatic", aria_hidden: "true" }
    更新日誌可以被自動識別嗎？

  %p
    非常困難，因為有各式各樣的提交訊息和檔案名稱難以完全掌握。


  %p
    #{link_to "Vandamme", data.links.vandamme} 是一個 Ruby 程式，由
    Gemnasium 團隊製作，可以解析多種（但絕對不是全部）開源庫的更新日誌。


  %h4#yanked
    %a.anchor{ href: "#yanked", aria_hidden: "true" }
    那些後來撤下的版本怎麼辦？

  %p
    因為重大漏洞或安全性問題而被撤下（unpublished）的版本通常不會出現在更新日誌中，但仍然建議記錄下來。你可以這樣作出記錄：

  %p <code>## [0.0.5] - 2014-12-13 [YANKED]</code>

  %p
    因為這類更改十分重要，所以 <code>[YANKED]</code> 標簽應該非常醒目，讓使用者注意到它是最重要的事。。此外，用方括號包圍可使其更易被程式識別。


  %h4#rewrite
    %a.anchor{ href: "#rewrite", aria_hidden: "true" }
    可以更改過去版本的日誌內容嗎？

  %p
    當然可以。總會有合適的原因去改進更新日誌。我也時常提
    Pull Request 來為那些未維護更新日誌的開源專案加入缺失的發佈信息。

  %p
    另外，你很有可能發現自己忘記記錄一個重大功能更新。這種情況下顯然應該重寫更新日誌。


  %h4#contribute
    %a.anchor{ href: "#contribute", aria_hidden: "true" }
    如何貢獻？

  %p
    本文檔並非<strong>真理</strong>。而是我經過深思熟慮、遵循蒐集到的資訊和範例之後提出的建議。

  %p
    源於我期望社群能達到共識，我相信討論的過程與結果一樣重要。

  %p
    所以歡迎<strong>#{link_to "貢獻", data.links.repo}</strong>。

.press
  %h3 訪談
  %p
    我在 #{link_to "The Changelog podcast", data.links.thechangelog}
    上講述了為什麼維護者與協作者應該在意更新日誌，以及建立這項專案背後的契機。
```

## File: test/build_test.rb
```ruby
require "minitest/autorun"

class TestMeme < Minitest::Test
  def setup
    @output = `bin/rake build`
  end

  def test_build
    assert @output.include?("Project built successfully.") 
  end
end
```
