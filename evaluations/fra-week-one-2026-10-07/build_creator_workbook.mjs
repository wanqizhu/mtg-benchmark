import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.dirname(new URL(import.meta.url).pathname);
const data = JSON.parse(await fs.readFile(path.join(root,'results/native-creators/workbook_data.json'),'utf8'));
const out = process.argv[2];
if (!out) throw new Error('Pass the final output workbook path');
const previews = path.join(root,'results/native-creators/previews');
await fs.mkdir(previews,{recursive:true});
const wb = Workbook.create();
const cards = wb.worksheets.add('Cards');
const models = wb.worksheets.add('Creator models');
const context = wb.worksheets.add('Context');
const ink = '#202837', header = '#273447', rule = '#D7DCE4';

function title(sheet,text,scope){
  sheet.showGridLines=false;
  sheet.getRange('A2').values=[[text]];
  sheet.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:ink};
  sheet.getRange('A2').format.rowHeight=25;
  sheet.getRange('A3').values=[[scope]];
  sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#566379'};
}
function writeTable(sheet,row,headers,values,col=0){
  const count=values.length+1,cols=headers.length;
  const range=sheet.getRangeByIndexes(row-1,col,count,cols);
  range.values=[headers,...values];
  range.format.font={name:'Arial',size:10,color:ink};
  range.format.rowHeight=22;
  range.format.verticalAlignment='center';
  range.format.columnWidth=20;
  const h=sheet.getRangeByIndexes(row-1,col,1,cols);
  h.format={fill:header,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,
            horizontalAlignment:'center',verticalAlignment:'center',rowHeight:38,
            borders:{insideVertical:{style:'thin',color:'#FFFFFF'}}};
  sheet.getRangeByIndexes(row,col,values.length,1).format.horizontalAlignment='left';
  if(cols>1)sheet.getRangeByIndexes(row,col+1,values.length,cols-1).format.horizontalAlignment='right';
  sheet.getRangeByIndexes(row-1,col,count,1).format.columnWidth=38;
  return range;
}

title(cards,'Creator card predictions','295 cards. Published grades and final fitted GIH conversions. October 7 observed snapshot.');
writeTable(cards,6,data.cards_header,data.cards);
cards.freezePanes.freezeRows(6);cards.freezePanes.freezeColumns(1);cards.tabColor=header;
cards.getRange('A7:A301').format.wrapText=true;
for(let i=0;i<10;i++){
  cards.getRangeByIndexes(6,1+i*2,295,1).format.horizontalAlignment='left';
  cards.getRangeByIndexes(6,1+i*2,295,1).format.wrapText=true;
  cards.getRangeByIndexes(6,2+i*2,295,1).setNumberFormat('0.00%');
}
for(let i=0;i<data.cards.length;i++){
  const longest=Math.max(...data.cards[i].filter((_,j)=>j%2===1).map(v=>String(v??'').length));
  cards.getRangeByIndexes(i+6,0,1,21).format.rowHeight=Math.max(22,18*Math.ceil(longest/21),18*Math.ceil(data.cards[i][0].length/35));
}
cards.getRange('A3').format.columnWidth=38;

title(models,'Creator models','Mapped GIH = native-grade curve + context offsets. Curves are fitted externally by monotone least squares.');
const summaryHeaders=['Creator','Native scale','Collected ratings','Mapped cards','Training cards','Grade rank correlation','Held-out MAE (pp)','Build-around offset (pp)','Included context terms'];
writeTable(models,6,summaryHeaders,data.model_summary);
models.getRange('B7:B16').format.horizontalAlignment='left';
models.getRange('B7:B16').format.wrapText=true;
models.getRange('A7:I16').format.rowHeight=32;
models.getRange('F7:H16').setNumberFormat('0.000');
models.getRange('I7:I16').format.horizontalAlignment='left';
models.getRange('A19').values=[['Final grade curves']];
models.getRange('A19').format.font={name:'Arial',size:12,bold:true,color:ink};
models.getRange('A20').values=[['Use the baseline for the selected grade, then add the listed offset when that creator labels the card a build-around.']];
models.getRange('A20').format.font={name:'Arial',size:10,italic:true,color:'#566379'};
writeTable(models,22,['Creator','Native grade labels','Native grade value','Cards at grade','Baseline GIH','Build-around offset (pp)'],data.mapping_rows);
models.getRangeByIndexes(22,4,data.mapping_rows.length,1).setNumberFormat('0.00%');
models.getRangeByIndexes(22,5,data.mapping_rows.length,1).setNumberFormat('0.000');
models.getRangeByIndexes(22,1,data.mapping_rows.length,1).format.horizontalAlignment='left';
models.getRange('A3').format.columnWidth=38;models.getRange('B6:B16').format.columnWidth=32;
models.freezePanes.freezeRows(6);models.freezePanes.freezeColumns(1);

title(context,'Contextual bucket counts','Distinct rated cards per creator. Only a bucket with more than 20 cards receives an extra regression term.');
writeTable(context,6,['Creator','Build-around cards','Sideboard cards','Archetype / matchup cards','Alternative grade cards','Included term','Training cards in included bucket'],data.bucket_summary);
context.getRange('F7:F16').format.horizontalAlignment='left';
context.getRange('I3').values=[['Original sources']];context.getRange('I3').format.font={name:'Arial',size:12,bold:true,color:ink};
writeTable(context,6,['Creator','Edition','Original source URL','Source caveat'],data.sources,8);
context.getRange('J7:L16').format.horizontalAlignment='left';
context.getRange('J6:J16').format.columnWidth=80;
context.getRange('K6:K16').format.columnWidth=92;
context.getRange('L6:L16').format.columnWidth=100;
for(const [row,text] of [[20,data.method],[22,data.grade_rule],[24,data.fit_rule]]){
  context.getRange(`A${row}`).values=[[text]];
  context.getRange(`A${row}`).format.font={name:'Arial',size:10,color:ink};
}
context.freezePanes.freezeRows(6);context.freezePanes.freezeColumns(1);

wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Cards!A6:G11',include:'values,formulas',tableMaxRows:6,tableMaxCols:7,maxChars:2600})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Creator models!A6:I16',include:'values',tableMaxRows:11,tableMaxCols:9,maxChars:3500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Final error scan'})).ndjson);
const contextualRow=data.cards.findIndex(row=>row[15]&&String(row[15]).length>21)+7;
const longestCardRow=data.cards.reduce((best,row,i)=>row[0].length>data.cards[best][0].length?i:best,0)+7;
for(const [sheetName,range,file] of [['Cards','A1:G13','cards'],['Cards',`A${longestCardRow}:E${longestCardRow+2}`,'long-card-names'],['Cards',`O${contextualRow}:U${contextualRow+3}`,'contextual-grades'],['Creator models','A1:I17','models'],['Context','A1:G17','context'],['Context','I3:L16','sources']]){
  const preview=await wb.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(previews,file+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(path.dirname(out),{recursive:true});
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(out);
await fs.copyFile(out,path.join(root,'results/native-creators/creator_predictions.xlsx'));
console.log('Saved '+out);
