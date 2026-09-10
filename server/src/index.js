import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import mongoose from 'mongoose';
import path from 'path';
import { fileURLToPath } from 'url';
import Activity from './models/Activity.js';
import Person from './models/Person.js';

const app = express();
const port = process.env.PORT || 5000;
const memoryActivities = [];
const memoryPeople = [];
let mongo = false;

app.use(helmet({ crossOriginResourcePolicy: false }));
app.use(cors({ origin: process.env.CLIENT_ORIGIN || 'http://localhost:5173' }));
app.use(express.json({ limit: '100kb' }));
app.use(morgan('tiny'));

if (process.env.MONGO_URI) {
  mongoose.connect(process.env.MONGO_URI)
    .then(() => { mongo = true; console.log('MongoDB connected'); })
    .catch((error) => console.warn('MongoDB unavailable; using memory:', error.message));
}

app.get('/api/health', (_, response) => response.json({ ok: true, storage: mongo ? 'mongodb' : 'memory' }));

app.get('/api/people', async (_, response) => {
  try {
    const people = mongo ? await Person.find().sort({ createdAt: -1 }).lean() : memoryPeople;
    response.json(people);
  } catch { response.status(500).json({ error: 'Could not load people' }); }
});

app.post('/api/people', async (request, response) => {
  const { name, details = '', faceDescriptor, consentGiven } = request.body;
  if (typeof name !== 'string' || !name.trim() || name.length > 80)
    return response.status(400).json({ error: 'A valid name is required' });
  if (consentGiven !== true)
    return response.status(400).json({ error: 'Explicit consent is required' });
  if (!Array.isArray(faceDescriptor) || faceDescriptor.length !== 128 || faceDescriptor.some((n) => !Number.isFinite(Number(n))))
    return response.status(400).json({ error: 'A valid 128-value face descriptor is required' });

  const clean = {
    name: name.trim(), details: String(details).trim().slice(0, 500),
    faceDescriptor: faceDescriptor.map(Number), consentGiven: true,
  };
  try {
    if (mongo) return response.status(201).json(await Person.create(clean));
    const saved = { ...clean, _id: crypto.randomUUID(), createdAt: new Date().toISOString() };
    memoryPeople.unshift(saved);
    response.status(201).json(saved);
  } catch { response.status(500).json({ error: 'Could not register person' }); }
});

app.delete('/api/people/:id', async (request, response) => {
  try {
    if (mongo) await Person.findByIdAndDelete(request.params.id);
    else {
      const index = memoryPeople.findIndex((person) => person._id === request.params.id);
      if (index >= 0) memoryPeople.splice(index, 1);
    }
    response.status(204).end();
  } catch { response.status(400).json({ error: 'Could not delete profile' }); }
});

app.get('/api/activities/recent', async (request, response) => {
  const limit = Math.min(Math.max(Number(request.query.limit) || 8, 1), 50);
  try {
    response.json(mongo ? await Activity.find().sort({ createdAt: -1 }).limit(limit).lean() : memoryActivities.slice(0, limit));
  } catch { response.status(500).json({ error: 'Could not load activity' }); }
});

app.post('/api/activities', async (request, response) => {
  const { label, kind, confidence, motionScore, objects = [], recognizedPersonName = '', recognizedPersonId = '' } = request.body;
  if (typeof label !== 'string' || !label.trim() || label.length > 160)
    return response.status(400).json({ error: 'Invalid label' });
  const clean = {
    label: label.trim(), kind: String(kind || 'unknown').slice(0, 32),
    confidence: Number(confidence) || 0, motionScore: Number(motionScore) || 0,
    objects: Array.isArray(objects) ? objects.slice(0, 20).map((value) => String(value).slice(0, 64)) : [],
    recognizedPersonName: String(recognizedPersonName).slice(0, 80),
    recognizedPersonId: String(recognizedPersonId).slice(0, 64),
  };
  try {
    if (mongo) return response.status(201).json(await Activity.create(clean));
    const saved = { ...clean, _id: crypto.randomUUID(), createdAt: new Date().toISOString() };
    memoryActivities.unshift(saved); memoryActivities.splice(100);
    response.status(201).json(saved);
  } catch { response.status(500).json({ error: 'Could not save activity' }); }
});

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dist = path.resolve(__dirname, '../../client/dist');
app.use(express.static(dist));
app.get('*', (request, response, next) => request.path.startsWith('/api/')
  ? next()
  : response.sendFile(path.join(dist, 'index.html'), (error) => error && next()));
app.listen(port, () => console.log(`ActivityLens API on http://localhost:${port}`));
