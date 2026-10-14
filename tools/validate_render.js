// Execute the report renderer with its actual JSON and a minimal DOM.
// This checks bindings and generated tables, not browser layout or CSS.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'results/report.html'), 'utf8');
const data = JSON.parse(fs.readFileSync(path.join(root, 'results/results.json'), 'utf8'));
const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], {textContent: '', innerHTML: ''}]));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
let fetched = false;
vm.runInNewContext(script, {
  console,
  document: {getElementById: id => elements.get(id)},
  fetch: async url => {
    assert.equal(url, 'results.json');
    fetched = true;
    return {ok: true, json: async () => data};
  },
});
setImmediate(() => {
  assert(fetched, 'renderer did not load data');
  assert.equal(elements.get('status').innerHTML, '', 'renderer reported an error');
  for (const [id, el] of elements) {
    if (id.endsWith('-table')) {
      assert(el.innerHTML.includes('<tbody>'), `table ${id} was not rendered`);
      assert(!/NaN|undefined/.test(el.innerHTML), `invalid value in ${id}`);
    }
    if (id.startsWith('arch-') && id !== 'arch-diagram') {
      assert(el.textContent, `diagram binding ${id} was not rendered`);
      assert(!/NaN|undefined/.test(el.textContent), `invalid value in ${id}`);
    }
  }
  console.log('Report JavaScript renders all tables and diagram bindings with current results.');
});
