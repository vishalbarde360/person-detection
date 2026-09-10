const readable = {
  'cell phone':'a phone', bottle:'a bottle', cup:'a cup', laptop:'a laptop',
  keyboard:'a keyboard', mouse:'a mouse', book:'a book', chair:'a chair'
};

export function describeActivity(predictions, motionScore, brightness) {
  const people = predictions.filter(p => p.class === 'person' && p.score > .52);
  const seen = predictions.filter(p => p.score > .5).map(p => p.class);
  if (brightness < 24) return {label:'Camera is dark or covered', kind:'warning', confidence:.9};
  if (!people.length) return {label:'No person detected', kind:'away', confidence:.82};
  if (people.length > 1) return {label:`${people.length} people are visible`, kind:'presence', confidence:Math.min(...people.map(p=>p.score))};
  if (seen.includes('cell phone')) return {label:'You may be using a phone', kind:'object', confidence:predictions.find(p=>p.class==='cell phone').score};
  if (seen.includes('cup') || seen.includes('bottle')) {
    const item=seen.includes('cup')?'cup':'bottle';
    return {label:`You are near ${readable[item]}`, kind:'object', confidence:predictions.find(p=>p.class===item).score};
  }
  const workObjects=['laptop','keyboard','mouse','book'].filter(x=>seen.includes(x));
  if (workObjects.length) return {label:`Working near ${workObjects.map(x=>readable[x]).join(' and ')}`, kind:'focus', confidence:.75};
  if (motionScore > 22) return {label:'You are moving actively', kind:'movement', confidence:.74};
  if (motionScore > 7) return {label:'Small movement detected', kind:'movement', confidence:.68};
  return {label:'You appear to be sitting still', kind:'still', confidence:.72};
}
