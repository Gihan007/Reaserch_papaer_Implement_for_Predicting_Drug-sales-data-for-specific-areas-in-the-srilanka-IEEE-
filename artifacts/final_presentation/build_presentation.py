from pathlib import Path
import json, math, zipfile, subprocess, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.xmlchemy import OxmlElement
from docx import Document
import imageio_ffmpeg

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
A = OUT/'assets'; A.mkdir(exist_ok=True)
METRICS = json.loads((ROOT/'src/evaluation_results/model_metrics.json').read_text())
SUMMARY = json.loads((ROOT/'src/evaluation_results/performance_summary.json').read_text())
ENS = json.loads((ROOT/'src/evaluation_results/ensemble_results.json').read_text())
STACK = json.loads((ROOT/'src/evaluation_results/stacking_results.json').read_text())
assert STACK['successful_categories'] == 8
NAMES = {'lightgbm':'LightGBM','lstm':'LSTM','gru':'GRU','xgboost':'XGBoost','transformer':'Transformer','sarimax':'SARIMAX','prophet':'Prophet'}
ORDER = sorted(NAMES, key=lambda m: SUMMARY[m]['final_mae'])
NAVY='102F4A'; BLUE='1678AD'; TEAL='008B8B'; PALE='EAF4F9'; WHITE='FFFFFF'; INK='17344A'; MUTED='526879'; GRAY='F4F7FA'; GOLD='BD7720'; LINE='D6E3EB'
prs = Presentation(); prs.slide_width=Inches(13.3333); prs.slide_height=Inches(7.5)
SLIDES=[]

def rect(s,x,y,w,h,fill=WHITE,line=None,radius=False):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill)
    if line:sh.line.color.rgb=RGBColor.from_string(line)
    else:sh.line.fill.background()
    if radius:sh.adjustments[0]=.12
    return sh

def txt(s,text,x,y,w,h,size=22,color=INK,bold=False,align=None):
    sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame
    tf.word_wrap=True;tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(str(text).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
        p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
        p.space_after=Pt(7)
        if align is not None:p.alignment=align
    return sh

def pic(s,path,x,y,w,h):
    with Image.open(path) as im: iw,ih=im.size
    scale=min(w/iw,h/ih);pw=iw*scale;ph=ih*scale
    return s.shapes.add_picture(str(path),Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))

def arrow(s,x1,y1,x2,y2):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=RGBColor.from_string(BLUE);sh.line.width=Pt(2)
    end=OxmlElement('a:tailEnd');end.set('type','triangle');sh.line._get_or_add_ln().append(end)

def card(s,x,y,w,h,kicker,body,fill=GRAY):
    rect(s,x,y,w,h,fill,radius=True);txt(s,kicker,x+.22,y+.18,w-.44,.55,23,BLUE,True);txt(s,body,x+.22,y+.87,w-.44,h-.95,21)

def callout(s,text,y=6.32,color=PALE):
    rect(s,.55,y,12.23,.55,color,radius=True);txt(s,text,.76,y+.10,11.8,.35,17,INK,True)

def base(title,section,seconds,notes,source='',backup=False):
    s=prs.slides.add_slide(prs.slide_layouts[6]);n=len(SLIDES)+1
    s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string(WHITE)
    rect(s,0,0,13.3333,.13,BLUE);txt(s,section.upper(),.55,.30,11,.28,12,BLUE,True)
    txt(s,title,.55,.80,12.22,.86,32,NAVY,True)
    rect(s,.55,7.05,12.23,.013,LINE)
    txt(s,'CS/2020/015  |  UNIVERSITY OF KELANIYA',.55,7.18,5,.2,9,MUTED)
    txt(s,('Q&A BACKUP' if backup else 'FINAL YEAR PROJECT  •  SEPTEMBER 2026'),5.2,7.18,6.6,.2,9,MUTED,align=PP_ALIGN.RIGHT)
    txt(s,f'{n:02}',12.2,7.12,.55,.3,14,BLUE,True,PP_ALIGN.RIGHT)
    if source:txt(s,'Source: '+source,.58,6.91,12,.13,7.8,MUTED)
    if backup:s._element.set('show','0')
    script=f"{'Q&A BACKUP — skip during the timed talk.' if backup else f'TARGET TIME: {seconds} seconds.'}\n\n{notes}\n\nEVIDENCE: {source}"
    s.notes_slide.notes_text_frame.text=script
    SLIDES.append(dict(number=n,title=title,section=section,seconds=seconds,notes=notes,source=source,backup=backup))
    return s

def table(s,headers,rows,x,y,w,row_h=.48,widths=None,font=18):
    widths=widths or [w/len(headers)]*len(headers)
    yy=y
    for r,row in enumerate([headers]+rows):
        xx=x
        for col,val in enumerate(row):
            rect(s,xx,yy,widths[col],row_h,NAVY if r==0 else (GRAY if r%2 else WHITE))
            txt(s,str(val),xx+.12,yy+.10,widths[col]-.23,row_h-.1,font,WHITE if r==0 else INK,r==0)
            xx+=widths[col]
        yy+=row_h

def make_demo():
    assets=[ROOT/'docs/thesisi version/chapter4_inserted_assets/figure_4_16_forecast_interface.png', ROOT/'services/frontend_service/app/static/images/C1_2018_01_15_xgboost_forecast.png',ROOT/'services/frontend_service/app/static/images/shap/fallback_importance_C1_xgboost_20260712_042311.png']
    titles=['01  Configure a request','02  Read a historical result','03  Inspect the explanation method']
    captions=['Choose a category, date and model in the project interface.', 'C1 request: 15 Jan 2018  |  nearest recorded week: 14 Jan 2018  |  sales: 28.33', 'Saved C1 XGBoost fallback output: sales_lag_4 ranks highest (0.2664).']
    font_path=Path('C:/Windows/Fonts/aptos.ttf')
    if not font_path.exists():font_path=Path('C:/Windows/Fonts/arial.ttf')
    f=ImageFont.truetype(str(font_path),32);fs=ImageFont.truetype(str(font_path),23)
    for i,path in enumerate(assets):
        canvas=Image.new('RGB',(1280,720),'#FFFFFF');d=ImageDraw.Draw(canvas)
        d.rectangle((0,0,1280,90),fill='#'+NAVY);d.text((35,25),titles[i],font=f,fill='white')
        with Image.open(path) as im:
            im=im.convert('RGB')
            if i==0: im=im.crop((0,0,im.width,690))
            im.thumbnail((1200,485),Image.Resampling.LANCZOS);canvas.paste(im,((1280-im.width)//2,100+(485-im.height)//2))
        d.rectangle((0,595,1280,720),fill='#'+PALE);d.text((30,613),captions[i],font=fs,fill='#'+INK)
        d.text((30,660),'OFFLINE WALKTHROUGH • saved project screens • no narration',font=fs,fill='#'+MUTED)
        canvas.save(A/f'demo_frame_{i}.png')
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    manifest=A/'demo_concat.txt'
    manifest.write_text(''.join(f"file '{(A/f'demo_frame_{i}.png').as_posix()}'\nduration 15\n" for i in range(3))+f"file '{(A/'demo_frame_2.png').as_posix()}'\n",encoding='utf-8')
    subprocess.run([ff,'-y','-f','concat','-safe','0','-i',str(manifest),'-t','45','-r','25','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(OUT/'project_walkthrough.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

make_demo()

s=base('Intelligent Drug Sales Forecasting', 'CSCI 43018 • Research Project',30,
"Good morning. I am Bope Ranasinghage Gihan Lakmal, CS/2020/015. My project is Intelligent Drug Sales Forecasting and Healthcare Insights with Artificial Intelligence, supervised by Professor N. G. J. Dias. I developed a category-wise forecasting application for pharmaceutical demand planning. I will explain the data, implemented methods, recorded results and practical application, followed by the main limitations.", 'Final_1 thesis, cover and Chapters 3–6')
txt(s,'and Healthcare Insights\nwith Artificial Intelligence',.6,1.70,11.8,1.4,34,NAVY,True)
rect(s,.60,3.32,8.0,.63,PALE,radius=True);txt(s,'Category-wise pharmaceutical demand planning in Sri Lanka',.8,3.48,7.6,.3,18,BLUE,True)
txt(s,'Bope Ranasinghage Gihan Lakmal',.6,4.55,7,.45,24,INK,True)
txt(s,'CS/2020/015',.6,5.1,5,.4,22,BLUE,True)
txt(s,'SUPERVISOR',9.0,4.5,3.6,.3,13,MUTED,True);txt(s,'Prof. N. G. J. Dias',9,4.96,3.7,.8,24,INK,True)
txt(s,'B.Sc. Honours in Computer Science\nFaculty of Computing and Technology • University of Kelaniya',.6,5.94,11.7,.65,18,MUTED)

s=base('From the research problem to the application','Overview',20,
"The presentation follows five stages: the problem and objectives; the dataset and methods; the application architecture; the experimental findings; and the demonstration, limitations and conclusion. The planned speaking time is under twenty minutes, including the embedded walkthrough. Detailed results and implementation notes are available in hidden backup slides for the panel discussion.")
for i,(a,b) in enumerate([('01','Problem & objectives'),('02','Data & methodology'),('03','System implementation'),('04','Results & explanation'),('05','Demonstration & conclusion')]):
    y=1.85+i*.86;rect(s,.6,y,.65,.58,BLUE,radius=True);txt(s,a,.65,y+.12,.55,.3,19,WHITE,True,PP_ALIGN.CENTER);txt(s,b,1.55,y+.08,10,.5,25,INK,True)

s=base('The problem: planning stock from changing sales','Problem',50,
"The project addresses pharmaceutical sales forecasting as a stock-planning problem. Too little stock can leave demand unmet; too much stock ties up working capital and can lead to expiry. Different drug categories have different demand levels and patterns, so a single forecasting method may not work equally well for every category. The research question is how to compare multiple forecasting approaches and make their outputs accessible through a usable application. The implemented forecasts use category sales history. Regional deployment is the intended application context; the bundled dataset does not contain separate region identifiers.", 'Thesis §§1.2–1.6; data/raw/C1.csv–C8.csv')
card(s,.6,1.95,3.86,3.55,'Understocking','Insufficient stock\nUnmet demand\nReactive purchasing')
card(s,4.74,1.95,3.86,3.55,'Overstocking','Capital tied up\nStorage pressure\nRisk of expiry')
card(s,8.88,1.95,3.86,3.55,'Forecasting challenge','Different category patterns\nLimited real data\nNeed for understandable outputs')
callout(s,'Research question: how can historical sales support usable, category-wise forecasting?')

s=base('Objectives and the delivered scope','Objectives',45,
"The main aim was to implement an AI-based drug sales forecasting system. I organised weekly sales into eight categories, implemented several model families, compared the seven models for which saved metrics are available, and exposed the forecasting and explanation functions through a web application. The research also contains experimental advanced-AI components. I distinguish those extensions from the core models with recorded comparative results. The practical contribution is the integrated forecasting workflow and category-level analysis, rather than a claim that a new forecasting algorithm has been invented.", 'Thesis §1.4; model_metrics.json; services/ and src/models/')
for i,(a,b) in enumerate([('Prepare','Organise and inspect category-wise weekly sales.'),('Predict','Implement statistical, boosted-tree and sequence models.'),('Compare','Assess MAE, RMSE, MAPE and recorded timing.'),('Deliver','Provide forecasts, plots and explanations in a web application.')]):
    y=1.85+i*1.06;txt(s,f'{i+1:02}',.65,y,.65,.5,28,BLUE,True);txt(s,a,1.55,y,2.05,.5,25,INK,True);txt(s,b,3.6,y+.03,8.95,.76,23)

s=base('Related work motivates a broader comparison','Literature',60,
"The directly related 2025 paper by Ekanayake and colleagues, including myself, used SARIMA with a Flask application for categorical drug sales. That provides the earlier project context. Ke and colleagues introduced LightGBM as an efficient gradient-boosted tree method, supporting its inclusion in the comparison. Hewamalage, Bergmeir and Bandara reviewed recurrent networks for time-series forecasting, motivating the LSTM and GRU comparisons. The present project extends the earlier category-wise application through additional model families, explanation support and modular services. I am not claiming that forecasting dashboards or model comparison are absent from all prior research.", 'Ekanayake et al. (2025), local paper; Ke et al. (2017) [2]; Hewamalage et al. (2021) [3]')
table(s,['Study','Relevant contribution','Role in this project'],[
['Ekanayake et al., 2025 [1]','SARIMA + Flask drug-sales application','Earlier categorical forecasting work'],
['Ke et al., 2017 [2]','Efficient gradient-boosted trees','LightGBM model comparison'],
['Hewamalage et al., 2021 [3]','Review of recurrent forecasting models','LSTM / GRU comparison']],.6,2.0,12.1,.98,[3.3,4.35,4.45],19)
callout(s,'Project extension: multi-model comparison + explanations + modular application services.')

s=base('The dataset: eight weekly sales series','Data',65,
"The dataset consists of eight category CSV files, each with 517 weekly observations, covering 12 January 2014 to 3 December 2023. The final data was described as an approximately equal mixture of real sales from two pharmacies in Kurunegala and synthetic data, which I confirmed when preparing these slides. The CSV files do not label individual rows by source, so exact real-versus-synthetic row counts cannot be independently separated. Across the eight files there are 4,136 category-week values, not 4,136 independent dates. The files have no missing cells or duplicate dates and all adjacent dates are seven days apart. There is no pharmacy or region column, so the measured results concern categories rather than regional generalisation.", 'data/raw/C1.csv–C8.csv; thesis §3.2; student confirmation, 13 Sep 2026')
for x,num,lab in [(0.6,'8','drug categories'),(3.73,'517','weeks per category'),(6.86,'2014–2023','stored date range'),(9.99,'50:50','real / synthetic*')]:
    rect(s,x,1.95,2.74,1.52,PALE,radius=True);txt(s,num,x+.18,2.17,2.4,.6,32,BLUE,True);txt(s,lab,x+.18,2.94,2.45,.3,17,MUTED)
txt(s,'12 Jan 2014 → 03 Dec 2023',.65,3.9,6.6,.45,25,INK,True)
txt(s,'One date + one sales value in each category file.\nNo missing cells; no duplicate dates; consistent 7-day spacing.',.65,4.55,7.25,1.3,21)
rect(s,8.45,3.94,4.25,1.75,GRAY,radius=True);txt(s,'Source context',8.7,4.13,3.8,.4,22,BLUE,True);txt(s,'Two Kurunegala pharmacies\nplus synthetic augmentation',8.7,4.67,3.7,.8,20)
callout(s,'*Student-confirmed approximate mix; row-level source labels and regional identifiers are unavailable.')

s=base('A model-specific preparation and forecasting pipeline','Methodology',70,
"Each CSV is parsed as an ordered time series. The input representation then depends on the model. XGBoost and LightGBM use five previous sales observations as lag features. LSTM, GRU and the standard Transformer use ten-observation sequences in their default paths. Statistical models use the dated series directly. LightGBM and the main sequence models apply MinMax scaling, and predictions are transformed back to the original sales scale. Model files and scalers are stored for reuse where supported. One current exception is LightGBM: its forecasting function retrains a small local model because saved native artifacts are not portable across all local builds. This slide shows the implemented preparation process; it does not assert that the archived benchmark used a verified common held-out split.", 'src/utils/preprocessing.py; src/models/{lightgbm,lstm,gru,transformer}_model.py')
for x,label,body in [(.6,'Load & order','Category CSV\nDate + sales'),(3.78,'Build inputs','5 lags / 10-step windows\nor dated series'),(6.96,'Train / load','Model + scaler\nCategory-specific paths'),(10.14,'Return output','Sales-scale result\nPlot + explanation')]:
    rect(s,x,2.15,2.58,2.23,PALE,radius=True);txt(s,label,x+.15,2.42,2.3,.7,23,BLUE,True);txt(s,body,x+.15,3.28,2.3,.95,18)
for x in [3.22,6.40,9.58]:arrow(s,x,3.28,x+.46,3.28)
txt(s,'Lag input example',.7,4.92,3,.4,22,BLUE,True)
txt(s,'[ y(t−1), y(t−2), y(t−3), y(t−4), y(t−5) ]  →  y(t)',3.8,4.9,8.5,.6,25,INK,True)
callout(s,'Implementation detail: LightGBM currently retrains locally during forecasting for artifact compatibility.')

s=base('Statistical and machine learning methods','Methodology',60,
"SARIMAX and Prophet provide statistical comparisons. The evaluator configures SARIMAX with order one-zero-zero and seasonal order one-zero-zero-seven. Because the input rows are weekly, that seasonal period is seven observations, not seven days. Prophet receives a date column named ds and target named y. The two machine-learning approaches are XGBoost and LightGBM, both using lagged sales values. XGBoost is a boosted-tree regressor. The current LightGBM function uses MinMax-normalised lags, 31 leaves, learning rate 0.05 and 100 boosting rounds. These are implementation defaults; I do not claim the separate optimisation JSON was used to produce every saved metric.", 'src/evaluation/model_evaluation.py; src/models/{sarimax,prophet,xgboost,lightgbm}_model.py')
card(s,.6,1.95,5.94,3.78,'Statistical models','SARIMAX: autoregressive / seasonal structure\nEvaluator: (1,0,0) × (1,0,0,7)\nProphet: dated trend and seasonality inputs')
card(s,6.8,1.95,5.94,3.78,'Boosted-tree models','XGBoost: regression on 5 lag features\nLightGBM: 5 normalised lag features\nCurrent defaults: 31 leaves, 0.05 learning rate, 100 boosting rounds')
callout(s,'Seasonal period 7 means seven observations in this weekly dataset; defaults are not tuned-run provenance.')

s=base('Deep learning uses sequences of past sales','Methodology',60,
"The standard LSTM and GRU implementations use two recurrent layers with 64 hidden units and dropout of 0.1. A linear output layer predicts the next sales value from the last hidden output. Their default input window contains ten weekly observations. The standard Transformer uses a model dimension of 64, four attention heads and two encoder layers, followed by a scalar prediction head. Predictions can be fed back recursively to obtain additional steps. The repository also contains TFT, N-BEATS and Informer implementations. However, those three do not have results in the saved seven-model comparison. N-BEATS is a separate forecasting architecture and should not be classified as a Transformer model.", 'src/models/{lstm,gru,transformer,tft,nbeats,informer}_model.py; model_metrics.json')
card(s,.6,1.95,3.87,3.76,'LSTM','10-observation windows\n2 recurrent layers\n64 hidden units\nDropout: 0.1')
card(s,4.73,1.95,3.87,3.76,'GRU','10-observation windows\n2 recurrent layers\n64 hidden units\nDropout: 0.1')
card(s,8.87,1.95,3.87,3.76,'Transformer','10-observation windows\nModel dimension: 64\n4 attention heads\n2 encoder layers')
callout(s,'TFT, N-BEATS and Informer are additional implementations; no comparable saved benchmark is reported.')

s=base('Stacking now has measured forecasting results','New stacking experiment',50,
"Stacking has now been implemented and evaluated using XGBoost, LSTM and GRU as base models, combined by StandardScaler and Ridge regression. Three expanding chronological blocks provide thirty out-of-fold prediction rows to train the combiner. For each category, the first 507 weeks are available for training, and the last ten weeks, from 1 October to 3 December 2023, are reserved for testing. Stacking achieved average MAE 30.2059 and RMSE 36.0614. It improved over the last-value naive baseline, but equal averaging and the individual bases were better on mean absolute error in this experiment. These are new results from a different protocol and must not be directly ranked against the thesis's original seven-model table.", 'src/evaluation_results/stacking_results.json; src/models/stacking_model.py; new run, 13 Sep 2026')
stack_labels={'stacking':'Stacking / Ridge','equal_average':'Equal average','naive_last':'Last-value baseline','xgboost':'XGBoost','lstm':'LSTM','gru':'GRU'}
table(s,['Method in the new experiment','Mean MAE','Mean RMSE'],[[stack_labels[m],f"{STACK['summary'][m]['MAE']:.4f}",f"{STACK['summary'][m]['RMSE']:.4f}"] for m in ['stacking','equal_average','xgboost','lstm','gru','naive_last']],.65,1.9,12,.49,[6,3,3],19)
txt(s,'XGBoost + LSTM + GRU forecasts  →  Ridge combiner  →  one result per future week',.8,5.7,11.8,.5,20,BLUE,True)
callout(s,'New, separate chronological test: stacking beats the naive baseline, but not simple averaging.')

s=base('A modular application connects users and models','Implementation',55,
"The current web application is served by FastAPI with Jinja templates and browser-side JavaScript. The frontend sends requests through the gateway. The repository separates forecasting, training, explanation and advanced-AI code into services. In the current gateway implementation, the forecast endpoint directly calls the shared forecasting function, so the diagram represents component responsibilities and not a claim that every request makes a network hop to a separate process. Saved models, scalers and CSV data support these components. Docker Compose and Kubernetes configuration are included, but this presentation does not claim a measured production deployment or a completed load test.", 'services/; apps/api_gateway/main.py; infra/docker/; infra/kubernetes/')
rect(s,4.68,1.88,4, .68,PALE,radius=True);txt(s,'Web UI • FastAPI / Jinja / JavaScript',4.78,2.08,3.8,.35,14,BLUE,True,PP_ALIGN.CENTER)
arrow(s,6.68,2.58,6.68,2.91);rect(s,4.68,2.96,4,.65,NAVY,radius=True);txt(s,'API gateway',4.9,3.12,3.55,.35,22,WHITE,True,PP_ALIGN.CENTER)
for x,label,body in [(.65,'Forecast','Prediction + plots'),(3.82,'Training','Model / scaler artifacts'),(6.99,'Explainability','SHAP / fallback'),(10.16,'Advanced AI','Experimental modules')]:
    arrow(s,6.68,3.65,x+1.27,4.02);rect(s,x,4.08,2.53,1.2,GRAY,radius=True);txt(s,label,x+.1,4.26,2.33,.42,21,BLUE,True,PP_ALIGN.CENTER);txt(s,body,x+.08,4.86,2.37,.3,14,INK,align=PP_ALIGN.CENTER)
rect(s,2.8,5.57,7.75,.47,PALE,radius=True);txt(s,'Shared CSV data • trained model files • scalers • result artifacts',3.02,5.68,7.3,.27,17,INK,True,PP_ALIGN.CENTER)
callout(s,'Component view: the current gateway forecast endpoint calls shared forecasting code directly.')

s=base('Evaluation: distinguish the original and new experiments','Evaluation',65,
"The presentation contains two separate experiments. The original saved benchmark compares seven models across eight categories and has a target-alignment and full-series fitting limitation. Its results remain labelled as archived. The new stacking experiment trains its base models and scalers only on earlier records, uses three chronological validation blocks to train Ridge, and reserves the final ten weeks as an untouched test. Stacking, equal averaging, the three bases and a naive baseline are compared on exactly the same test dates in this new run. Both sets use mean category MAE and RMSE, but their different training and evaluation procedures prevent direct cross-table ranking. MAPE is undefined if a held-out actual is zero.", 'src/evaluation/model_evaluation.py; model_metrics.json; performance_summary.json')
for x,label,body in [(.6,'MAE','Average absolute error\nLower is better'),(4.73,'RMSE','Emphasises large errors\nLower is better'),(8.86,'MAPE','Percentage error\nSensitive to small actuals')]:card(s,x,1.93,3.88,2.05,label,body)
txt(s,'Original benchmark + new stacking holdout',.7,4.36,11.8,.55,27,BLUE,True)
txt(s,'Overall values = arithmetic mean of the eight category metrics.',.7,5.08,11.8,.6,22)
callout(s,'New stacking test: 507 training weeks + 10 untouched test weeks. Original benchmark limitations remain.',color='FFF2DF')

s=base('Original benchmark: LightGBM leads on absolute error','Results',65,
"This chart orders the seven models by average MAE. LightGBM has the lowest recorded MAE of 23.3728 and RMSE of 28.9163, followed by LSTM and GRU on both absolute-error metrics. XGBoost has an average MAE of 25.4679. SARIMAX and Prophet have larger absolute errors in these saved results. The choice of metric matters: Transformer has the lowest average MAPE at 53.6691 percent, even though it is not best on MAE or RMSE. Because category scales differ and the evaluation has the limitation described on the previous slide, these figures are descriptive comparisons of the archived experiment, not a guarantee of future performance.", 'performance_summary.json; values checked against category means in model_metrics.json')
for i,m in enumerate(ORDER):
    yy=1.9+i*.53;val=SUMMARY[m]['final_mae'];txt(s,NAMES[m],.65,yy+.07,2.05,.34,20,INK,m=='lightgbm')
    rect(s,2.8,yy,5.35*(val/35),.36,TEAL if m=='lightgbm' else BLUE);txt(s,f'{val:.4f}',2.92+5.35*(val/35),yy+.02,1.2,.35,18,INK,True)
txt(s,'Mean category MAE • original sales scale • lower is better',.7,5.93,8.3,.3,15,MUTED)
rect(s,9.22,1.93,3.5,3.68,PALE,radius=True);txt(s,'LightGBM',9.48,2.2,2.98,.5,25,BLUE,True);txt(s,'23.3728',9.48,3.0,3,.65,34,NAVY,True);txt(s,'average MAE',9.48,3.65,3,.4,18,MUTED);txt(s,'28.9163',9.48,4.26,3,.6,30,NAVY,True);txt(s,'average RMSE',9.48,4.95,3,.4,18,MUTED)
callout(s,'Original experiment only: these numbers are not directly comparable to the new stacking test.')

s=base('Original benchmark: the best model varies by category','Results',55,
"The category-level comparison shows why the application supports model choice. GRU has the lowest recorded MAE for C1. Prophet is lowest for C2 and C3, LightGBM for C4, LSTM for C5, C6 and C8, and XGBoost for C7. That means LSTM leads in three categories while LightGBM leads on average. C4 has a much larger error scale than C6, so an average absolute error can be influenced by higher-volume categories. Selecting a model from the same results used to judge it is descriptive; a production selection rule would require separate validation and later testing. These category winners do not establish the best model for an unobserved pharmacy or region.", 'src/evaluation_results/model_metrics.json; minimum MAE within each category')
rows=[]
for c,ms in METRICS.items():
    best=min(ms,key=lambda m:ms[m]['MAE']);rows.append([c,NAMES[best],f'{ms[best]["MAE"]:.4f}'])
table(s,['Category','Lowest recorded MAE model','MAE'],rows,.65,1.8,8.0,.49,[1.45,4.65,1.9],18)
rect(s,9,2.02,3.68,3.76,PALE,radius=True);txt(s,'No universal winner',9.22,2.23,3.25,.85,27,BLUE,True);txt(s,'LSTM leads in 3 of 8 categories.\n\nLightGBM leads on average MAE and RMSE.',9.22,3.34,3.16,2.2,20)
callout(s,'A deployment model-selection rule still needs an independent validation and test period.')

s=base('Explainability must name the method actually returned','Explainability',55,
"The explanation service attempts SHAP-based interpretation and also provides a fallback. This saved example is explicitly labelled fallback_lag_importance. Its top feature is sales_lag_4 with a normalised importance of approximately 0.2664. The fallback uses model feature importance where available, otherwise normalised absolute correlations with the target. It therefore gives a relative ranking of lag features. It does not establish that a particular lag caused demand, and it is not equivalent to a signed local SHAP explanation of an individual prediction. The application should report which method was returned so users can interpret the chart correctly.", 'Saved fallback_importance_C1_xgboost_20260712_042311.png; shap_explainer.py §§411–458; thesis §4.4')
pic(s,ROOT/'services/frontend_service/app/static/images/shap/fallback_importance_C1_xgboost_20260712_042311.png',.62,1.88,8.0,4.22)
rect(s,8.9,1.95,3.8,4.12,PALE,radius=True);txt(s,'Saved C1 example',9.15,2.18,3.25,.55,24,BLUE,True);txt(s,'sales_lag_4',9.15,3.05,3.25,.55,26,NAVY,True);txt(s,'0.2664',9.15,3.65,3.25,.62,34,NAVY,True);txt(s,'normalised importance\n\nMethod: fallback\nlag importance',9.15,4.42,3.1,1.55,18)
callout(s,'This output ranks lag importance; it is neither a causal effect nor a local SHAP attribution.')

s=base('Advanced AI modules are experimental extensions','Research extensions',55,
"Beyond the core forecast comparison, the repository contains advanced research modules. A saved C1 neural architecture search evaluated eight architectures. Its losses are on a scaled series and cannot be directly compared with the sales-scale MAE table. A recorded federated test used two simulated clients and one FedAvg round; this is not a field trial across independent pharmacies. Meta-learning and adaptation workflows are implemented, but the recorded status says the MAML model was not trained at that point. The causal-analysis output includes correlations with time features; this does not establish causal effects. These components demonstrate implemented research workflows, without a verified comparative accuracy gain for the main forecast benchmark.", 'nas_results_C1_1783854913.json; fed_results_C1_iid_1783854914.json; causal_discovery_C1.json; thesis §§4.5,4.8')
table(s,['Module','Available evidence','Interpretation'],[
['Neural architecture search','C1 run: 8 architectures evaluated','Small search; scaled metrics'],
['Federated learning','2 simulated clients; 1 FedAvg round','Local demonstration'],
['Meta-learning','Adaptation code; recorded MAML untrained','No demonstrated benchmark gain'],
['Causal analysis','Recorded temporal-feature correlations','Association, not causal proof']],.62,2.0,12.1,.8,[3.25,4.68,4.17],19)
callout(s,'Implemented extension does not imply independently validated improvement in forecasting accuracy.')

s=base('Application demonstration • embedded offline walkthrough','Application',85,
"I will now play a 45-second silent walkthrough assembled from the project interface and saved outputs. First, the interface allows the user to choose a category, date and forecasting method. Second, the recorded historical example requests C1 on 15 January 2018 and returns the nearest stored weekly observation, 14 January 2018, with sales of 28.33. This is a historical lookup, not a prediction of an unseen value. Third, the explanation example shows the saved fallback importance chart. The film uses archived screens and does not show a fresh live backend run. The following hidden backup slide includes the same screens if video playback is unavailable. Click the video to start; it has no audio.", 'Project UI screenshot; C1 historical output chart; saved C1 fallback explanation; 45-second silent MP4')
s.shapes.add_movie(str(OUT/'project_walkthrough.mp4'),Inches(2.166),Inches(1.66),Inches(9),Inches(5.0625),poster_frame_image=str(A/'demo_frame_0.png'),mime_type='video/mp4')
txt(s,'Click to play • 45 seconds • saved screens, historical lookup and explanation',.65,6.75,12,.2,11,MUTED,align=PP_ALIGN.CENTER)

s=base('Stacking passes functional and chronological checks','Verification',40,
"The updated implementation passed sixteen Python tests, covering the existing service smoke checks and new stacking checks, plus one JavaScript UI test. Tests verify that future holdout values do not enter model fitting, each horizon is combined separately, weekly date conversion is correct, and missing or stale artifacts fail explicitly. All eight full-history model bundles were also loaded and generated finite ten-week forecasts. A real gateway request for C1 on 11 December 2023 returned approximately 53.0036 for the corresponding weekly endpoint, 17 December. The stacking interface now uses actual historical values and recorded held-out scores, rather than the old placeholder metrics and charts. This integration check is separate from the unseen-future evaluation.", 'tests/test_stacking.py; tests/test_services_smoke.py; tests/test_stacking_ui.cjs; stacking_integration_checks.json')
card(s,.6,1.96,3.88,3.78,'17 checks passed','16 Python tests\n1 JavaScript UI test\nChronology + horizon checks\nArtifact integrity + errors')
card(s,4.73,1.96,3.88,3.78,'Stacking API check','C1 request: 11 Dec 2023\nWeekly end: 17 Dec 2023\nForecast: 53.0036\nHTTP status: 200')
card(s,8.86,1.96,3.88,3.78,'Application output','Actual recorded sales chart\nMeasured holdout errors\nNo invented accuracy score\nUnsupported intervals omitted')
callout(s,'All 8 stacking artifacts produce finite forecasts. The separate holdout measures forecasting error.')

s=base('Limitations define the next research steps','Limitations & future work',65,
"There are four main limitations. First, the dataset contains synthetic augmentation and has neither source labels nor region identifiers, so future work needs traceable real data from more pharmacies. Second, the original archived evaluator still needs a consistent chronological design. The new stacking experiment now uses an untouched ten-week test and chronological meta-training, but that fix does not retroactively validate the older results. Third, stacking now converts dates to weekly horizons and limits predictions to its trained ten-step horizon. Older model routes still need the same date alignment correction. Fourth, low-volume series make percentage errors unstable. The full C6 series contains 46 zeros, although this does not mean all are in the evaluation window. Future evaluation should add suitable low-volume metrics and simple naive baselines. Real inventory impact has not yet been measured.", 'CSV audit; src/evaluation/model_evaluation.py; forecast_service/app/forecasting.py; thesis §§5.6,6.7')
table(s,['Current limitation','Next step'],[
['Hybrid data; no source / regional labels','Collect traceable, region-labelled pharmacy histories'],
['Original benchmark horizon / split issues','New stacking is aligned; repeat for the other models'],
['Older API routes use day counts as steps','Stacking uses weeks; extend the fix to older routes'],
['Low sales destabilise percentage error','Add suitable scaled errors and inspect low-volume cases']],.65,1.98,12,.8,[5.6,6.4],20)
callout(s,'Operational stock-out reduction, cost savings and real multi-pharmacy generalisation remain unmeasured.')

s=base('Conclusion: a working platform with clear research findings','Conclusion',45,
"The project delivered a category-wise forecasting platform with a web interface, model comparison, stored artifacts and explanation support. In the saved seven-model comparison, LightGBM has the lowest average MAE and RMSE. Category-wise results show that the lowest-error model changes across categories. Explainability is useful when its returned method and limits are explicit. The new stacking experiment now has a consistent future holdout: its mean MAE is 30.2059, compared with 24.2382 for equal averaging. The next priority is broader real-data evaluation, followed by testing the system in pharmacy operations. The system provides a foundation for demand planning, while actual reductions in shortages or wastage remain future outcomes to measure.", 'Thesis Chapters 4–6; saved evaluation artifacts and implementation audit')
for i,(a,b) in enumerate([('Delivered','A category-wise forecasting and explanation application.'),('Observed','Stacking works; simple averaging is better in the new holdout.'),('Next','Strengthen future-period validation and real pharmacy evaluation.')]):
    y=2+i*1.35;rect(s,.65,y,.08,.98,TEAL);txt(s,a,1,y,2.55,.5,26,BLUE,True);txt(s,b,3.8,y+.01,8.6,1.0,25)

s=base('Selected references','References',10,
"These are the key sources cited in the talk. Reference one is the directly related categorical drug-sales paper stored with the project. References two and three support the selected machine-learning and recurrent modelling methods. The exact project figures come from the thesis, source code and saved JSON files identified in the notes and accompanying evidence report.", 'Local project paper; NeurIPS proceedings; International Journal of Forecasting; final thesis and repository')
refs=[('[1]','S. B. Ekanayake, M. Nasmeen, G. Lakmal, P. Vimanshani and A. Perera. “Predicting medical drug sales in a specific area for categorical drugs using time series forecasting.” SCSE, 2025. Local project copy.'),('[2]','G. Ke et al. “LightGBM: A Highly Efficient Gradient Boosting Decision Tree.” Advances in Neural Information Processing Systems 30, 2017, pp. 3146–3154.'),('[3]','H. Hewamalage, C. Bergmeir and K. Bandara. “Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions.” International Journal of Forecasting, 37(1), 2021, pp. 388–427.'),('[4]','B. R. G. Lakmal. Final-year thesis, CS/2020/015, University of Kelaniya, 2026; accompanying source code, category CSVs and saved evaluation JSON files.')]
for i,(n,body) in enumerate(refs):
    y=1.9+i*1.15;txt(s,n,.65,y,.65,.5,21,BLUE,True);sh=txt(s,body,1.4,y,11.2,1.03,18)
    if i in [1,2]:sh.text_frame.paragraphs[0].runs[0].hyperlink.address=['','https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html','https://doi.org/10.1016/j.ijforecast.2020.06.008'][i]

s=base('Thank you','Acknowledgement & questions',10,
"Thank you to Professor N. G. J. Dias for his guidance, to the Faculty of Computing and Technology, and to the people who supported the project. Thank you for listening. I welcome the panel's questions. Hidden backup slides contain exact metrics, model settings, category mapping, evaluation details and the offline demonstration screens.")
txt(s,'Questions & discussion',.7,2.47,11.9,1.0,44,BLUE,True,PP_ALIGN.CENTER)
txt(s,'Bope Ranasinghage Gihan Lakmal  •  CS/2020/015',.7,3.9,11.9,.6,25,INK,True,PP_ALIGN.CENTER)
txt(s,'With thanks to Prof. N. G. J. Dias\nand the Faculty of Computing and Technology, University of Kelaniya',1.5,5.05,10.3,1,21,MUTED,align=PP_ALIGN.CENTER)

s=base('Complete recorded model metrics','Backup A',0,'Use this table for numerical questions. The errors and timings are arithmetic averages across eight categories. Timing is taken from the archived evaluator; it includes path-specific loading and computation and is not a controlled current-production latency benchmark. The source data is preserved at four decimals. Different metrics give different rankings.', 'src/evaluation_results/performance_summary.json',True)
table(s,['Model','MAE','RMSE','MAPE (%)','Time (s)'],[[NAMES[m]]+[f'{SUMMARY[m][k]:.4f}' for k in ['final_mae','final_rmse','final_mape','final_time']] for m in ORDER],.65,1.98,12,.51,[3.2,2.2,2.2,2.2,2.2],18)
callout(s,'XGBoost has the lowest recorded mean time (0.0343 s); current paths and hardware may differ.')

s=base('Implementation defaults and category mapping','Backup B',0,'These settings were read from current source files. They are not asserted to be the exact configurations behind all archived metrics. XGBoost and LightGBM use five lags; the standard LSTM, GRU and Transformer use ten-step windows. The category-to-code mapping is checked by comparing all values in each C1–C8 file with the corresponding column in the combined CSV. The codes identify drug categories; they are not regional labels.', 'src/models/; data/raw/pharama weekly sales copy.csv and C1.csv–C8.csv',True)
table(s,['Model','Current default'],[['XGBoost','5 lags; squared-error objective; 100 rounds'],['LightGBM','5 lags; 31 leaves; lr 0.05; 100 rounds'],['LSTM / GRU','10 steps; 2 layers; hidden 64; dropout 0.1'],['Transformer','10 steps; d=64; 4 heads; 2 encoder layers']],.65,1.93,12,.66,[3.3,8.7],18)
txt(s,'C1 → M01AB    C2 → M01AE    C3 → N02BA    C4 → N02BE\nC5 → N05B      C6 → N05C       C7 → R03         C8 → R06',.8,5.48,11.8,1.2,23,BLUE,True)

s=base('How to interpret the evaluation limitation','Backup C',0,'The thesis describes chronological train–test separation. The available evaluator sets actual_values to the last ten observations, fits SARIMAX on the complete series and requests predictions over its final ten indices. Several other functions instead read the full CSV and predict beyond its end. That compares different effective horizons and does not prove an unseen-future holdout. Correct evaluation should choose a cutoff, fit preprocessing and each model using only earlier observations, generate the same ten future weekly timestamps, and score against aligned actuals. Repeat across origins if data allows. The presentation does not invent a split percentage or claim a rerun.', 'src/evaluation/model_evaluation.py lines 42–80; model-specific forecast functions',True)
card(s,.65,1.96,5.87,3.85,'Available evaluator','Targets: final 10 actual observations\nSome predictions: beyond the full series\nSARIMAX: fit on the full series\nResult: no common verified holdout')
card(s,6.8,1.96,5.87,3.85,'Required validation design','Choose a chronological cutoff\nFit models / scalers on earlier data\nForecast the same future weeks\nCompare aligned actuals; repeat origins')
callout(s,'Keep the archived results as recorded evidence; rerun a corrected protocol before stronger accuracy claims.')

s=base('Dataset profiles: different scales and demand patterns','Backup D',0,'These are descriptive plots of the bundled CSV values, not predictions. Each panel uses its own sales axis. The full dataset has 46 zero values in C6; this should not be confused with the number of zeros in any evaluation window. Source mix is student-confirmed at approximately 50:50. Exact row provenance remains unavailable.', 'data/raw/C1.csv, C4.csv, C6.csv',True)
fig,axes=plt.subplots(3,1,figsize=(12,5.1),sharex=True)
for ax,c in zip(axes,['C1','C4','C6']):
    d=pd.read_csv(ROOT/f'data/raw/{c}.csv',parse_dates=['datum']);ax.plot(d.datum,d[c],color='#'+BLUE,lw=.9);ax.set_ylabel(c+' sales',fontsize=10);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
axes[-1].set_xlabel('Stored weekly dates');fig.tight_layout();fig.savefig(A/'dataset_profiles.png',dpi=180);plt.close(fig)
pic(s,A/'dataset_profiles.png',.65,1.83,12.02,4.77)

s=base('Demonstration fallback: saved project screens','Backup E',0,'Use these static images if the embedded MP4 cannot play. The interface is a saved project screen. The chart shows the historical C1 lookup: requested date 2018-01-15, nearest observation 2018-01-14, sales 28.33. This is not a future forecast or an accuracy test. The saved importance chart is available on the main explainability slide.', 'Project UI screenshot and C1_2018_01_15_xgboost_forecast.png',True)
pic(s,ROOT/'docs/thesisi version/chapter4_inserted_assets/figure_4_16_forecast_interface.png',.6,1.9,6.0,4.5)
pic(s,ROOT/'services/frontend_service/app/static/images/C1_2018_01_15_xgboost_forecast.png',6.85,1.9,5.87,4.5)
callout(s,'Saved historical lookup: C1 → 28.33 on 14 Jan 2018; this demonstrates retrieval and presentation.')

# Prevent accidental automatic slide advancement. The video itself is click-to-play.
for s in prs.slides:
    tr=OxmlElement('p:transition');tr.set('advClick','1');s._element.insert_element_before(tr,'p:timing','p:extLst')
prs.core_properties.title='Intelligent Drug Sales Forecasting and Healthcare Insights with Artificial Intelligence'
prs.core_properties.subject='CSCI 43018 final-year presentation'
prs.core_properties.author='Bope Ranasinghage Gihan Lakmal'
prs.core_properties.keywords='CS/2020/015; University of Kelaniya; pharmaceutical sales forecasting'
prs.save(OUT/'CS_2020_015.pptx')
(OUT/'slide_manifest.json').write_text(json.dumps(SLIDES,indent=2,ensure_ascii=False),encoding='utf-8')

doc=Document();doc.add_heading('CS/2020/015 — Presentation speaker guide',0)
total=sum(x['seconds'] for x in SLIDES)
doc.add_paragraph(f'Planned delivery: {total//60}:{total%60:02d}, including a 45-second silent walkthrough. 22 main slides and 5 hidden Q&A backup slides. No recorded narration.')
doc.add_paragraph('Rehearse aloud and finish within 20 minutes. Keep the references brief. The timings are a speaking plan, not automatic slide transitions. Click the video to play on slide 17; slide 27 provides a static fallback.')
doc.add_paragraph('Submission: CS_2020_015.pptx, by 18 September 2026. Use the supervisor-reviewed version for evaluation. This file has not been submitted or represented as supervisor-approved.')
elapsed=0
for item in SLIDES:
    doc.add_heading(f"{item['number']:02}. {item['title']}",level=1)
    if item['backup']:doc.add_paragraph('Hidden Q&A backup — outside the timed presentation.')
    else:
        end=elapsed+item['seconds'];doc.add_paragraph(f"Target: {elapsed//60}:{elapsed%60:02d}–{end//60}:{end%60:02d} ({item['seconds']} seconds)");elapsed=end
    doc.add_paragraph(item['notes']);doc.add_paragraph('Evidence: '+item['source'])
doc.add_heading('Panel preparation',level=1)
qa=[('What is the main contribution?','An integrated category-wise forecasting application, now including a functioning chronological stack and a separate future-period evaluation. Avoid claiming a newly invented forecasting algorithm.'),('Is all the data real?','No. The student confirmed an approximately 50:50 real/synthetic mix. Real-data context: two Kurunegala pharmacies. Row labels and the generation recipe are not preserved in the category CSVs.'),('Does this predict a specific region?','The application targets Sri Lankan pharmacy planning, but the bundled evaluation data is category-wise and has no region identifier. Regional generalisation has not been established.'),('What train–test split was used?','The original benchmark has alignment limitations. For the new stacking experiment, 507 weeks are pre-test history and the final 10 are untouched test data. Three earlier ten-week blocks train the Ridge combiner using out-of-fold predictions.'),('Why LightGBM?','It has the lowest recorded average MAE and RMSE among the seven compared models. This is an empirical description, not proof of superiority on new data.'),('Is the ensemble best?','The original ensemble comparison did not beat LightGBM. In the new, separate aligned test, stacking MAE is 30.2059 and equal-average MAE is 24.2382. The old null entries remain archived; new results are in stacking_results.json.'),('Is the explanation SHAP?','The shown output is fallback_lag_importance. It can use model importance or normalised absolute correlations. Do not call it a local SHAP explanation or causal effect.'),('Was federated learning used across real pharmacies?','The recorded demonstration used two simulated clients and one FedAvg round. It is not a real distributed field evaluation.'),('What does the embedded video prove?','It illustrates the existing UI and archived outputs. The numerical example is a historical lookup. It is not a newly recorded live test or a forecast-accuracy experiment.'),('What would you improve first?','Trace data provenance, align weekly forecast horizons, rerun chronological or rolling-origin evaluation with simple baselines, and test on separately collected real pharmacy data.')]
for q,a in qa:doc.add_heading(q,level=2);doc.add_paragraph(a)
try:
    doc.save(OUT/'Speaker_Notes_CS_2020_015.docx')
except PermissionError:
    doc.save(OUT/'Speaker_Notes_CS_2020_015_Updated.docx')

# Structural and numeric checks without executing or changing project models.
issues=[]
for s in prs.slides:
    for sh in s.shapes:
        if sh.left<0 or sh.top<0 or sh.left+sh.width>prs.slide_width+100 or sh.top+sh.height>prs.slide_height+100:issues.append(f'Off-slide: {s.slide_id}, {sh.name}')
for m in NAMES:
    for metric,key in [('MAE','final_mae'),('RMSE','final_rmse'),('MAPE','final_mape')]:
        actual=np.mean([METRICS[c][m][metric] for c in METRICS]);expected=SUMMARY[m][key]
        assert abs(actual-expected)<=.000051,(m,metric,actual,expected)
combined=pd.read_csv(ROOT/'data/raw/pharama weekly sales copy.csv')
for i,col in enumerate(combined.columns[1:],1):assert np.allclose(combined[col],pd.read_csv(ROOT/f'data/raw/C{i}.csv')[f'C{i}'])
with zipfile.ZipFile(OUT/'CS_2020_015.pptx') as z:
    assert len([n for n in z.namelist() if n.endswith('.mp4')])>=1
assert not issues,issues
print(json.dumps({'pptx':str(OUT/'CS_2020_015.pptx'),'main_slides':22,'hidden_backup_slides':5,'planned_seconds':total,'planned_minutes':f'{total//60}:{total%60:02d}','numeric_check':'passed','boundary_check':'passed','embedded_video':'45 seconds, H.264, no audio'},indent=2))
