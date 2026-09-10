import mongoose from 'mongoose';

const schema = new mongoose.Schema(
  {
    label: { type: String, required: true, maxlength: 160 },
    kind: { type: String, maxlength: 32 },
    confidence: { type: Number, min: 0, max: 1 },
    motionScore: { type: Number, min: 0, max: 255 },
    objects: [{ type: String, maxlength: 64 }],
    recognizedPersonName: { type: String, maxlength: 80, default: '' },
    recognizedPersonId: { type: String, maxlength: 64, default: '' },
  },
  { timestamps: true, versionKey: false },
);

export default mongoose.model('Activity', schema);
