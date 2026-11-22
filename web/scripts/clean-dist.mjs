// web/scripts/clean-dist.mjs
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const frontendDir = path.resolve(__dirname, '../../frontend');
const assetsDir = path.join(frontendDir, 'assets');
const indexHtml = path.join(frontendDir, 'index.html');
const cssDir = path.join(frontendDir, 'css');
const jsDir = path.join(frontendDir, 'js');

console.log('[clean-dist] Starting safe distribution cleanup...');

// 1. Remove frontend/index.html
if (fs.existsSync(indexHtml)) {
  fs.unlinkSync(indexHtml);
  console.log('[clean-dist] Removed frontend/index.html');
}

// 2. Remove files inside frontend/assets/ (keep folder, preserve sample_leaf.jpg if needed)
if (fs.existsSync(assetsDir)) {
  const files = fs.readdirSync(assetsDir);
  for (const file of files) {
    const fullPath = path.join(assetsDir, file);
    try {
      const stat = fs.statSync(fullPath);
      if (stat.isDirectory()) {
        fs.rmSync(fullPath, { recursive: true, force: true });
      } else {
        fs.unlinkSync(fullPath);
      }
      console.log(`[clean-dist] Cleaned asset: ${file}`);
    } catch (err) {
      console.warn(`[clean-dist] Warning removing ${file}:`, err.message);
    }
  }
} else {
  fs.mkdirSync(assetsDir, { recursive: true });
}

// 3. Ensure frontend/css and frontend/js exist with .gitkeep
for (const dir of [cssDir, jsDir]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const gitkeep = path.join(dir, '.gitkeep');
  if (!fs.existsSync(gitkeep)) {
    fs.writeFileSync(gitkeep, '');
    console.log(`[clean-dist] Created .gitkeep in ${path.basename(dir)}`);
  }
}

console.log('[clean-dist] Safe distribution cleanup completed successfully.');
