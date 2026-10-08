const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      value: '', hidden: false, disabled: false, textContent: '',
      classList: {toggle() {}},
      addEventListener(type, listener) { this[type] = listener; },
      showModal() { this.open = true; },
      close() { this.open = false; }
    });
  }
  return elements.get(id);
}

const context = {
  document: {getElementById: element},
  fetch: async (path) => path.startsWith('/log')
    ? {ok: true, json: async () => ({entries: []})}
    : {ok: false, status: 429, headers: {get: () => '300'},
       json: async () => ({error: 'Submission rate limit exceeded'})}
};
vm.runInNewContext(fs.readFileSync('static/app.js', 'utf8'), context);

(async () => {
  element('creator-id').value = 'demo';
  element('creative-text').value = 'A brief sentence.';
  await element('submission-form').submit({preventDefault() {}});
  assert.equal(element('rate-limit-popup').open, true);
  assert.match(element('rate-limit-popup-message').textContent, /300 seconds/);
  assert.equal(element('analyze-button').disabled, false);
  element('rate-limit-popup-close').click();
  assert.equal(element('rate-limit-popup').open, false);
})().catch((error) => { console.error(error); process.exitCode = 1; });
