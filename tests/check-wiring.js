#!/usr/bin/env node
/**
 * 网页前端的静态接线检查（不需要浏览器，秒级完成）。
 *
 * 专治「改了 UI 结构但忘了改取值」这类只在运行时炸的错误：
 *   1. 脚本里 $('xxx') / getElementById('xxx') 取的每个 id，HTML 里必须真的存在；
 *   2. 脚本里 classList.add/remove/toggle('yyy') 用到的每个类名，样式表里必须有规则
 *      （没有规则 = 用户看不到这个状态 = 等于没有这个状态）；
 *   3. 页面引用的每个本地资源必须存在；
 *   4. 页面里不能有裸露的模板占位符 / undefined / NaN 文案。
 *
 * 这几条都能抓到真 bug，而且比浏览器测试快几百倍，适合每次改完都跑。
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const HTML_PATH = path.join(ROOT, 'spotlight', 'web', 'index.html');
const html = fs.readFileSync(HTML_PATH, 'utf8');

let pass = 0, fail = 0;
const fails = [];
function chk(ok, label) {
  if (ok) { pass++; console.log('  ✓ ' + label); }
  else { fail++; fails.push(label); console.log('  ✗ ' + label); }
}

// 分离 <style> 与 <script>
const styles = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/gi)].map(m => m[1]).join('\n');
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]).join('\n');
const bodyMarkup = html.replace(/<style[\s\S]*?<\/style>/gi, '').replace(/<script[\s\S]*?<\/script>/gi, '');

// ---- 1. id 接线
const htmlIds = new Set([...bodyMarkup.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
const usedIds = new Set([
  ...[...scripts.matchAll(/\$\('([^']+)'\)/g)].map(m => m[1]),
  ...[...scripts.matchAll(/getElementById\('([^']+)'\)/g)].map(m => m[1]),
]);
const missingIds = [...usedIds].filter(id => !htmlIds.has(id));
chk(htmlIds.size > 0, `页面定义了 ${htmlIds.size} 个 id`);
chk(usedIds.size > 0, `脚本引用了 ${usedIds.size} 个 id`);
chk(missingIds.length === 0,
  missingIds.length ? `脚本引用了不存在的 id: ${missingIds.join(', ')}` : '脚本引用的每个 id 在页面上都存在');

// ---- 2. class 接线（脚本加的类，样式表里要有规则）
const usedClasses = new Set([
  ...[...scripts.matchAll(/classList\.(?:add|remove|toggle)\('([^']+)'/g)].map(m => m[1]),
  ...[...scripts.matchAll(/className\s*=\s*'([^']+)'/g)].flatMap(m => m[1].split(/\s+/)),
].filter(Boolean));
const missingClasses = [...usedClasses].filter(c => !new RegExp(`\\.${c.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&')}\\b`).test(styles));
chk(usedClasses.size > 0, `脚本操作了 ${usedClasses.size} 个类名`);
chk(missingClasses.length === 0,
  missingClasses.length ? `脚本用了但样式表没有规则的类: ${missingClasses.join(', ')}` : '脚本操作的每个类名样式表里都有规则');

// ---- 3. 本地资源存在
const assets = [...html.matchAll(/<(?:script|link)\b[^>]*?\b(?:src|href)="([^"]+)"/gi)]
  .map(m => m[1])
  .filter(v => !/^(https?:|\/\/|#|data:)/.test(v));
const missingAssets = assets.filter(a => !fs.existsSync(path.join(ROOT, 'spotlight', 'web', a)));
// 本页刻意做成单文件（样式和脚本都内联），所以「0 个外部资源」是正常的，不做数量断言
console.log(`  · 页面引用了 ${assets.length} 个外部本地资源（单文件设计下为 0 是正常的）`);
chk(missingAssets.length === 0,
  missingAssets.length ? `引用了不存在的资源: ${missingAssets.join(', ')}` : '页面引用的本地资源都存在');

// ---- 3.5 UI flow and accessible controls
const workflowSteps = [...bodyMarkup.matchAll(/data-step="([^"]+)"/g)].map(m => m[1]);
chk(workflowSteps.join(',') === 'upload,select,settings,result', '处理流程按导入→选人→设置→导出排列');
chk(/id="drop"[^>]*role="button"[^>]*tabindex="0"/.test(bodyMarkup)
  && /\$\('drop'\)\.addEventListener\('keydown'/.test(scripts), '导入区域支持键盘触发');
chk(/<details class="advanced-settings">/.test(bodyMarkup)
  && /<summary>精细调节/.test(bodyMarkup), '高级效果默认折叠，基础操作更聚焦');
chk(/role="progressbar"[^>]*aria-valuenow="0"/.test(bodyMarkup)
  && /setAttribute\('aria-valuenow'/.test(scripts), '处理进度具备可访问读数');
chk(/data-mode="point" class="on" aria-pressed="true"/.test(bodyMarkup)
  && /setAttribute\('aria-pressed'/.test(scripts), '选择工具可读出选中状态');

// ---- 4. 文案卫生
chk(!/\$\{[a-z_]+\}/i.test(bodyMarkup), '页面文案里没有未替换的模板占位符');
chk(!/>\s*undefined\s*</.test(bodyMarkup), '页面文案里没有裸露的 undefined');
chk(!/lorem ipsum/i.test(html), '没有残留的占位文案');

// ---- 5. 接口路径与后端一致（防止前端调了一个后端没有的路由）
const serverSrc = fs.readFileSync(path.join(ROOT, 'spotlight', 'server.py'), 'utf8');
const routes = new Set([...serverSrc.matchAll(/@app\.(?:get|post|delete)\("([^"]+)"\)/g)].map(m => m[1]));
const called = new Set([...scripts.matchAll(/api\(\s*[`'"]([^`'"?]+)/g)].map(m => m[1])
  .concat([...scripts.matchAll(/fetch\(\s*[`'"]([^`'"?]+)/g)].map(m => m[1])));
const norm = r => r.replace(/\$\{[^}]+\}/g, '*').replace(/\/$/, '');
const routeNorms = [...routes].map(norm);
const missingRoutes = [...called].filter(c => {
  const n = norm(c);
  return !routeNorms.some(r => r === n || r.replace(/\{[^}]+\}/g, '*') === n);
});
chk(called.size > 0, `脚本调用了 ${called.size} 个接口`);
chk(missingRoutes.length === 0,
  missingRoutes.length ? `前端调用了后端没有的接口: ${missingRoutes.join(', ')}` : '前端调用的每个接口后端都有');

console.log('');
console.log((fail ? '❌' : '✅') + ` 接线检查：${pass} 通过 / ${fail} 失败`);
if (fail) { console.log('失败项：'); fails.forEach(f => console.log('  · ' + f)); }
process.exit(fail ? 1 : 0);
