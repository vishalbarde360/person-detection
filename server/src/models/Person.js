import mongoose from 'mongoose';

const schema = new mongoose.Schema(
  {
    name: { type: String, required: true, trim: true, maxlength: 80 },
    details: { type: String, default: '', maxlength: 500 },
    faceDescriptor: {
      type: [Number],
      required: true,
      validate: {
        validator: (value) => Array.isArray(value) && value.length === 128,
        message: 'A 128-value face descriptor is required',
      },
    },
    consentGiven: { type: Boolean, required: true },
  },
  { timestamps: true, versionKey: false },
);

export default mongoose.model('Person', schema);
