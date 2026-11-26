# Engineering Graph Explorer

Use Obsidian Graph for global topology and local graph for a selected note. Use the required Breadcrumbs plugin's Matrix and Tree panes to inspect the direction and type of engineering relations. Typed relations remain in note frontmatter and are validated by `edac`.

If Matrix or Tree views are missing, install and enable Breadcrumbs, then confirm that `.obsidian/plugins/breadcrumbs/data.json` is present. Do not replace typed relation fields with plain links as a workaround.

Native Graph edges require quoted wikilinks inside typed frontmatter fields. Run `edac doctor` and then `edac format --relation-links wikilink` if an EDAC-valid repository appears orphaned. Use the connected notes in `docs/_examples/` to check the standard Need → Requirement → Design → Verification colors, then remove the fixture directory.

For a deterministic local projection:

```bash
edac view NODE-ID --depth 2
```
