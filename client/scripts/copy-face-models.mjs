import fs from 'fs';
import path from 'path';

const candidates = [
  path.resolve('node_modules/@vladmandic/face-api/model'),
  path.resolve('node_modules/@vladmandic/face-api/dist/model'),
];
const source = candidates.find((candidate) => fs.existsSync(candidate));
const target = path.resolve('public/models');

if (!source) {
  console.warn('Face model files were not found in the package. See README.md.');
  process.exit(0);
}

fs.mkdirSync(target, { recursive: true });
const prefixes = ['tiny_face_detector', 'ssd_mobilenetv1', 'face_landmark_68', 'face_recognition'];
for (const file of fs.readdirSync(source)) {
  if (prefixes.some((prefix) => file.startsWith(prefix))) {
    fs.copyFileSync(path.join(source, file), path.join(target, file));
  }
}
console.log(`Copied face-recognition model files to ${target}`);
