import subprocess, sys
def _pip(*p): subprocess.check_call([sys.executable,'-m','pip','install','-q',*p])
_pip('gradio>=4.26','python-jobspy','plotly','pandas','numpy','scikit-learn','requests','PyPDF2','python-docx')

# ─────────────────────────────────────────────────────────────────────────────
#  Skill Gap Analyser — v2  (O*NET CSV + full pipeline)
#
#  Pipeline:
#   Resume → Keyword NLP (freq→0-5) → Skill Vector
#   O*NET CSV  → Role Requirement Vectors
#   Cosine Similarity → Top Role Matches
#   Threshold Gap Logic → Marginal Gain Ranking → Recommendations
#   JobSpy (LinkedIn + Indeed)  → Live Job Listings
#   Gradio UI  → Charts (Plotly)  → CSV Export
# ─────────────────────────────────────────────────────────────────────────────

import os, io, json, re, warnings, requests, urllib.request, zipfile, csv
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
SCRAPE_COUNTRY = 'India'
MAX_JOBS       = 10
CACHE_FILE = 'jobs_cache.json'

ONET_DIR = Path("onet_data")
ONET_URL       = 'https://www.onetcenter.org/dl_files/database/db_29_0_text.zip'

# ── Skill columns (63) ────────────────────────────────────────────────────────
SKILL_COLS = [
    'python','java','c','cpp','rust','javascript','typescript','sql','r','scala',
    'mysql','postgresql','mongodb','sqlite','data_modeling','data_warehousing','etl',
    'machine_learning','deep_learning','feature_engineering','neural_networks',
    'nlp','computer_vision','recommendation_systems','time_series','reinforcement_learning',
    'linear_algebra','probability_statistics','optimization','discrete_math',
    'data_structures_algorithms','competitive_programming','data_analysis','data_visualization',
    'devops','aws','gcp','azure','docker','linux','shell_scripting',
    'pandas','numpy','matplotlib','seaborn','html','css','react','angular',
    'nodejs','expressjs','rest_apis','git','jupyter','vscode',
    'tensorflow','pytorch','keras','scikit_learn','opencv','spacy','huggingface',
    'oop','testing_debugging','android_dev','ios_dev',
]

SKILL_LABELS = {
    'python':'Python','java':'Java','c':'C','cpp':'C++','rust':'Rust',
    'javascript':'JavaScript','typescript':'TypeScript','sql':'SQL','r':'R','scala':'Scala',
    'mysql':'MySQL','postgresql':'PostgreSQL','mongodb':'MongoDB','sqlite':'SQLite',
    'data_modeling':'Data Modeling','data_warehousing':'Data Warehousing','etl':'ETL',
    'machine_learning':'Machine Learning','deep_learning':'Deep Learning',
    'feature_engineering':'Feature Engineering','neural_networks':'Neural Networks',
    'nlp':'NLP','computer_vision':'Computer Vision',
    'recommendation_systems':'Recommender Systems','time_series':'Time Series',
    'reinforcement_learning':'Reinforcement Learning','linear_algebra':'Linear Algebra',
    'probability_statistics':'Probability & Stats','optimization':'Optimization',
    'discrete_math':'Discrete Math','data_structures_algorithms':'DSA',
    'competitive_programming':'Competitive Prog.','data_analysis':'Data Analysis',
    'data_visualization':'Data Visualization','devops':'DevOps','aws':'AWS',
    'gcp':'GCP','azure':'Azure','docker':'Docker','linux':'Linux',
    'shell_scripting':'Shell Scripting','pandas':'Pandas','numpy':'NumPy',
    'matplotlib':'Matplotlib','seaborn':'Seaborn','html':'HTML','css':'CSS',
    'react':'React','angular':'Angular','nodejs':'Node.js','expressjs':'Express.js',
    'rest_apis':'REST APIs','git':'Git/GitHub','jupyter':'Jupyter','vscode':'VS Code',
    'tensorflow':'TensorFlow','pytorch':'PyTorch','keras':'Keras',
    'scikit_learn':'Scikit-learn','opencv':'OpenCV','spacy':'spaCy',
    'huggingface':'HuggingFace','oop':'OOP','testing_debugging':'Testing & Debugging',
    'android_dev':'Android Dev','ios_dev':'iOS Dev',
}

SKILL_GROUPS = {
    '🖥️ Programming Languages': ['python','java','c','cpp','rust','javascript','typescript','sql','r','scala'],
    '🗄️ Databases':             ['mysql','postgresql','mongodb','sqlite','data_modeling','data_warehousing','etl'],
    '🤖 ML / AI':               ['machine_learning','deep_learning','feature_engineering','neural_networks',
                                  'nlp','computer_vision','recommendation_systems','time_series','reinforcement_learning'],
    '📐 Math / CS':             ['linear_algebra','probability_statistics','optimization','discrete_math',
                                  'data_structures_algorithms','competitive_programming'],
    '📊 Data':                  ['data_analysis','data_visualization'],
    '☁️ DevOps / Cloud':        ['devops','aws','gcp','azure','docker','linux','shell_scripting'],
    '📦 Libraries & Frameworks':['pandas','numpy','matplotlib','seaborn','html','css','react','angular',
                                  'nodejs','expressjs','rest_apis','git','jupyter','vscode',
                                  'tensorflow','pytorch','keras','scikit_learn','opencv','spacy','huggingface'],
    '⚙️ General':               ['oop','testing_debugging','android_dev','ios_dev'],
}

# ═════════════════════════════════════════════════════════════════════════════
#  O*NET INTEGRATION
#  Uses free O*NET database CSVs (db_29_0_text.zip from onetcenter.org).
#  Skills mapped via O*NET Element IDs → our SKILL_COLS vocabulary.
#  Falls back to built-in ROLE_VECTORS if CSV download fails.
# ═════════════════════════════════════════════════════════════════════════════

# O*NET SOC codes → our role names (hand-curated mapping)
ONET_SOC_MAP = {
    '15-1252.00': 'Software Engineer',
    '15-1253.00': 'Software Engineer',           # Software Quality Assurance
    '15-1299.08': 'Backend Developer',
    '15-1254.00': 'Backend Developer',
    '15-1255.00': 'Frontend Developer',
    '15-1299.09': 'Full Stack Developer',
    '15-1211.01': 'Mobile App Developer',
    '15-1299.07': 'Android Developer',
    '15-1299.06': 'iOS Developer',
    '17-2061.00': 'Embedded / Systems Engineer',
    '15-2051.00': 'Data Scientist',
    '15-2051.01': 'Data Scientist',
    '15-2031.00': 'Data Analyst',
    '15-1243.00': 'Data Engineer',
    '15-1244.00': 'Data Engineer',
    '15-2051.02': 'Machine Learning Engineer',
    '15-1299.04': 'AI Engineer',
    '15-1299.05': 'NLP Engineer',
    '15-1299.03': 'Computer Vision Engineer',
    '15-1245.00': 'DevOps Engineer',
    '15-1241.00': 'Cloud Engineer',
    '15-1244.01': 'MLOps Engineer',
    '13-2051.00': 'Quantitative Analyst',
    '15-1299.02': 'Research Scientist (AI/ML)',
    '15-1243.01': 'BI Developer',
    '13-1111.00': 'Business Analyst',
    '15-1251.00': 'DevOps Engineer',
    '15-1212.00': 'Site Reliability Engineer',
    '15-1299.10': 'Cybersecurity Engineer',
    '15-1241.01': 'Site Reliability Engineer',
    '25-1021.00': 'Research Scientist (AI/ML)',
}

# O*NET Technology Skills element titles → our skill keys (substring match)
ONET_TECH_KEYWORDS = {
    'python':                  ['python'],
    'java':                    ['java'],
    'cpp':                     ['c++'],
    'javascript':              ['javascript'],
    'typescript':              ['typescript'],
    'sql':                     ['sql','structured query'],
    'r':                       [' r ','r programming','rstudio'],
    'scala':                   ['scala'],
    'rust':                    ['rust'],
    'mysql':                   ['mysql'],
    'postgresql':              ['postgresql','postgres'],
    'mongodb':                 ['mongodb'],
    'machine_learning':        ['machine learning','scikit','sklearn'],
    'deep_learning':           ['deep learning','neural network'],
    'tensorflow':              ['tensorflow'],
    'pytorch':                 ['pytorch'],
    'keras':                   ['keras'],
    'nlp':                     ['natural language','nlp'],
    'computer_vision':         ['computer vision','opencv'],
    'docker':                  ['docker','kubernetes'],
    'aws':                     ['amazon web services','aws'],
    'gcp':                     ['google cloud','gcp'],
    'azure':                   ['microsoft azure','azure'],
    'linux':                   ['linux','unix'],
    'git':                     ['git','github'],
    'react':                   ['react'],
    'angular':                 ['angular'],
    'nodejs':                  ['node.js','nodejs'],
    'html':                    ['html'],
    'css':                     ['css'],
    'data_analysis':           ['data analysis','analytics'],
    'data_visualization':      ['tableau','power bi','data visualization'],
    'devops':                  ['devops','continuous integration'],
}

# O*NET Knowledge/Skills element → our skill keys
ONET_KSA_KEYWORDS = {
    'python':                  ['programming','scripting'],
    'sql':                     ['database','data management'],
    'machine_learning':        ['machine learning','artificial intelligence'],
    'data_analysis':           ['data analysis','statistics'],
    'probability_statistics':  ['mathematics','statistics'],
    'linear_algebra':          ['mathematics'],
    'data_structures_algorithms': ['algorithms','computer science'],
    'oop':                     ['software design','systems design'],
    'testing_debugging':       ['quality assurance','testing'],
    'rest_apis':               ['web services'],
    'linux':                   ['operating systems'],
    'devops':                  ['systems administration'],
}


def _download_onet():
    """Download and extract O*NET database CSVs to ONET_DIR. Returns True on success."""
    ONET_DIR.mkdir(parents=True, exist_ok=True)
    needed = ['Technology Skills.txt', 'Knowledge.txt', 'Skills.txt',
              'Occupation Data.txt', 'Task Statements.txt']
    if all((ONET_DIR / f).exists() for f in needed):
        return True
    print('Downloading O*NET database (~80 MB)…')
    try:
        zip_path = ONET_DIR / 'onet.zip'
        urllib.request.urlretrieve(ONET_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.namelist():
                fname = Path(member).name
                if fname in needed:
                    with z.open(member) as src, open(ONET_DIR / fname, 'wb') as dst:
                        dst.write(src.read())
        zip_path.unlink(missing_ok=True)
        print('O*NET data ready.')
        return True
    except Exception as e:
        print(f'O*NET download failed: {e}  →  using built-in role vectors.')
        return False


def _build_onet_vectors():
    """
    Parse O*NET CSVs and build role requirement vectors aligned to SKILL_COLS.
    Returns dict: {role_name: {skill: score 0-5}}
    """
    try:
        tech_df = pd.read_csv(ONET_DIR / 'Technology Skills.txt', sep='\t',
                              usecols=['O*NET-SOC Code','Example'], dtype=str).fillna('')
        occ_df  = pd.read_csv(ONET_DIR / 'Occupation Data.txt', sep='\t',
                              usecols=['O*NET-SOC Code','Title'], dtype=str).fillna('')
    except Exception as e:
        print(f'O*NET parse error: {e}')
        return None

    # Build role→tech-skill counts
    role_tech: dict[str, dict[str, int]] = {}
    for _, row in tech_df.iterrows():
        soc   = str(row['O*NET-SOC Code']).strip()
        role  = ONET_SOC_MAP.get(soc)
        if not role:
            continue
        tech_name = str(row['Example']).lower()
        if role not in role_tech:
            role_tech[role] = {}
        for sk, keywords in ONET_TECH_KEYWORDS.items():
            if any(kw in tech_name for kw in keywords):
                role_tech[role][sk] = role_tech[role].get(sk, 0) + 1

    # Convert counts → 0-5 scale using min-capped formula
    onet_vectors = {}
    for role, tech_counts in role_tech.items():
        if not tech_counts:
            continue
        vec = {}
        max_c = max(tech_counts.values()) if tech_counts else 1
        for sk in SKILL_COLS:
            raw = tech_counts.get(sk, 0)
            # min-capped sum formula: scale to 1-5, 0 if absent
            vec[sk] = round(min(raw / max_c * 5, 5), 1) if raw > 0 else 0
        onet_vectors[role] = vec

    print(f'O*NET loaded')
    return onet_vectors if onet_vectors else None


# ── Fallback built-in role vectors (used if O*NET unavailable) ────────────────
_BUILTIN_ROLE_VECTORS = {
    'Software Engineer':           {'python':4,'java':4,'data_structures_algorithms':5,'oop':5,'git':4,'testing_debugging':4,'sql':3,'rest_apis':3,'linux':3},
    'Backend Developer':           {'python':4,'java':3,'nodejs':3,'sql':4,'rest_apis':4,'docker':3,'linux':3,'git':4,'aws':3,'testing_debugging':3,'oop':4,'postgresql':3,'mongodb':3},
    'Frontend Developer':          {'javascript':5,'typescript':4,'react':5,'html':5,'css':5,'angular':3,'nodejs':3,'git':4,'rest_apis':3,'testing_debugging':3},
    'Full Stack Developer':        {'javascript':5,'react':4,'nodejs':4,'python':3,'sql':3,'rest_apis':4,'html':5,'css':4,'git':4,'docker':3,'mongodb':3},
    'Mobile App Developer':        {'android_dev':4,'ios_dev':4,'javascript':3,'react':3,'git':4,'rest_apis':3,'java':3},
    'Android Developer':           {'android_dev':5,'java':4,'rest_apis':3,'git':4,'sql':2,'oop':4},
    'iOS Developer':               {'ios_dev':5,'git':4,'rest_apis':3,'oop':4,'testing_debugging':3},
    'Embedded / Systems Engineer': {'c':5,'cpp':5,'linux':4,'rust':3,'data_structures_algorithms':4,'oop':3,'shell_scripting':3},
    'Machine Learning Engineer':   {'python':5,'machine_learning':5,'deep_learning':4,'tensorflow':3,'pytorch':3,'scikit_learn':4,'numpy':4,'pandas':4,'data_analysis':4,'linear_algebra':4,'probability_statistics':4,'feature_engineering':4,'sql':3,'git':3},
    'AI Engineer':                 {'python':5,'deep_learning':5,'pytorch':4,'tensorflow':4,'nlp':4,'machine_learning':5,'huggingface':4,'neural_networks':4,'git':3,'docker':3},
    'Data Scientist':              {'python':5,'machine_learning':4,'data_analysis':5,'data_visualization':4,'sql':4,'pandas':5,'numpy':4,'scikit_learn':4,'probability_statistics':5,'linear_algebra':3,'jupyter':4,'git':3},
    'Data Analyst':                {'sql':5,'data_analysis':5,'data_visualization':5,'python':4,'pandas':4,'probability_statistics':3,'git':2},
    'Data Engineer':               {'python':4,'sql':5,'etl':5,'data_warehousing':4,'data_modeling':4,'aws':3,'docker':3,'linux':3,'git':4,'postgresql':3,'mongodb':2},
    'NLP Engineer':                {'python':5,'nlp':5,'deep_learning':4,'huggingface':4,'pytorch':4,'tensorflow':3,'probability_statistics':3,'linear_algebra':3,'git':3},
    'Computer Vision Engineer':    {'python':5,'computer_vision':5,'deep_learning':4,'pytorch':4,'tensorflow':3,'opencv':4,'numpy':4,'linear_algebra':4,'git':3},
    'MLOps Engineer':              {'python':4,'machine_learning':4,'docker':5,'devops':5,'aws':4,'gcp':3,'linux':4,'git':4,'shell_scripting':3},
    'Quantitative Analyst':        {'python':4,'r':3,'probability_statistics':5,'linear_algebra':5,'optimization':4,'data_analysis':4,'sql':3,'time_series':4,'machine_learning':3},
    'Research Scientist (AI/ML)':  {'python':5,'deep_learning':5,'pytorch':4,'tensorflow':3,'nlp':3,'computer_vision':3,'linear_algebra':5,'probability_statistics':5,'optimization':4,'data_analysis':4},
    'DevOps Engineer':             {'docker':5,'linux':5,'aws':4,'shell_scripting':4,'python':3,'git':4,'devops':5,'gcp':3,'azure':3},
    'Cloud Engineer':              {'aws':5,'gcp':4,'azure':4,'docker':4,'linux':4,'python':3,'devops':4,'shell_scripting':3},
    'Site Reliability Engineer':   {'linux':5,'docker':4,'python':4,'devops':5,'aws':4,'shell_scripting':4,'git':4},
    'Cybersecurity Engineer':      {'linux':5,'python':3,'shell_scripting':4,'docker':3,'git':3,'testing_debugging':4},
    'Business Analyst':            {'data_analysis':5,'sql':4,'data_visualization':4,'probability_statistics':3,'python':2},
    'BI Developer':                {'sql':5,'data_visualization':5,'data_warehousing':4,'data_modeling':4,'python':3,'etl':4},
    'Game Developer':              {'cpp':5,'python':3,'data_structures_algorithms':4,'oop':5,'linear_algebra':4,'git':3},
    'Blockchain Developer':        {'javascript':4,'python':3,'data_structures_algorithms':4,'oop':4,'rest_apis':3,'git':4},
}

ROLE_VECTORS       = _BUILTIN_ROLE_VECTORS.copy()   # may be replaced by O*NET at startup
ONET_AVAILABLE     = False

def _init_onet():
    global ROLE_VECTORS, ONET_AVAILABLE

    if not ONET_DIR.exists():
        raise Exception("O*NET data folder not found. Please add it manually.")

    onet_vecs = _build_onet_vectors()

    if onet_vecs:
        # Merge: O*NET data takes precedence; supplement gaps from built-in
        merged = _BUILTIN_ROLE_VECTORS.copy()

        for role, vec in onet_vecs.items():
            if role in merged:
                # blend: 60% O*NET + 40% builtin for roles in both
                blended = {}
                for sk in SKILL_COLS:
                    o = vec.get(sk, 0)
                    b = merged[role].get(sk, 0)
                    blended[sk] = round(0.6 * o + 0.4 * b, 2)
                merged[role] = blended
            else:
                merged[role] = vec

        ROLE_VECTORS = merged
        ONET_AVAILABLE = True
        print("O*NET data loaded successfully ✅")

    else:
        print("Using built-in role vectors ⚠️")
        ROLE_VECTORS = _BUILTIN_ROLE_VECTORS
        ONET_AVAILABLE = False

_init_onet()

PREFERRED_ROLES = list(ROLE_VECTORS.keys()) + ['Research / Academic Track', 'Undecided / Exploring']

# ═════════════════════════════════════════════════════════════════════════════
#  RESUME PARSING
# ═════════════════════════════════════════════════════════════════════════════

RESUME_KEYWORDS = {sk: kws for sk, kws in {
    'python':['python'],'java':['java'],'c':[' c ','c programming'],'cpp':['c++','cpp'],
    'javascript':['javascript',' js '],'typescript':['typescript'],'sql':['sql','mysql','postgresql'],
    'machine_learning':['machine learning','ml ','scikit','sklearn'],
    'deep_learning':['deep learning','tensorflow','keras','pytorch'],
    'nlp':['natural language','nlp','spacy','nltk'],'docker':['docker','kubernetes'],
    'aws':['aws','amazon web services'],'react':['react'],'nodejs':['node.js','nodejs'],
    'git':['git','github'],'linux':['linux','ubuntu'],'data_analysis':['data analysis','pandas'],
    'tensorflow':['tensorflow'],'pytorch':['pytorch'],'huggingface':['huggingface','transformers'],
    'feature_engineering':['feature engineering'],'probability_statistics':['statistics','probability'],
    'linear_algebra':['linear algebra'],'data_structures_algorithms':['data structures','algorithms'],
    'devops':['devops','ci/cd','jenkins'],'angular':['angular'],'mongodb':['mongodb'],
    'postgresql':['postgresql','postgres'],'rest_apis':['rest api','restful'],
    'html':['html'],'css':['css'],'scikit_learn':['scikit'],
}.items()}

PROJECT_SKILL_PHRASES = {
    'python':['python','django','flask','fastapi'],
    'javascript':['javascript','node','express','jquery'],
    'typescript':['typescript'],'java':['java','spring boot'],'cpp':['c++','cpp'],
    'sql':['sql','mysql','postgresql','sqlite','relational database'],
    'mongodb':['mongodb','mongoose'],'postgresql':['postgresql','postgres'],
    'machine_learning':['machine learning','ml model','classification','regression',
                        'random forest','xgboost','scikit','sklearn'],
    'deep_learning':['deep learning','neural network','cnn','rnn','lstm','transformer','bert'],
    'nlp':['natural language','nlp','text classification','sentiment analysis','named entity',
           'tokenization','word embedding','bert','spacy','nltk','huggingface'],
    'computer_vision':['computer vision','image classification','object detection','yolo','opencv'],
    'recommendation_systems':['recommendation','recommender','collaborative filtering'],
    'time_series':['time series','forecasting','arima','prophet'],
    'data_analysis':['data analysis','exploratory data','eda','data cleaning','pandas','numpy'],
    'data_visualization':['visualization','dashboard','tableau','power bi','matplotlib','seaborn','plotly'],
    'tensorflow':['tensorflow'],'pytorch':['pytorch'],'keras':['keras'],
    'scikit_learn':['scikit','sklearn'],'huggingface':['huggingface','transformers library'],
    'opencv':['opencv','cv2'],'spacy':['spacy'],
    'react':['react','reactjs','next.js','hooks','redux'],
    'angular':['angular'],'nodejs':['node.js','nodejs','express.js'],
    'html':['html','html5'],'css':['css','css3','tailwind','bootstrap'],
    'rest_apis':['rest api','restful','api endpoints','fastapi','flask api'],
    'docker':['docker','dockerfile','kubernetes','k8s'],
    'aws':['aws','amazon web services','s3','ec2','lambda','sagemaker'],
    'gcp':['gcp','google cloud','bigquery','vertex ai'],
    'azure':['azure','microsoft azure'],
    'linux':['linux','ubuntu','bash script','shell script'],
    'devops':['devops','ci/cd','github actions','jenkins','terraform'],
    'git':['git','github','gitlab','version control'],
    'data_structures_algorithms':['data structure','algorithm','dynamic programming','graph algorithm'],
    'oop':['object-oriented','oop','design pattern','solid principle'],
    'testing_debugging':['unit test','pytest','jest','tdd','test driven','debugging'],
    'android_dev':['android','kotlin','android studio'],'ios_dev':['ios','swift','xcode'],
    'etl':['etl','data pipeline','airflow'],'data_warehousing':['data warehouse','snowflake','redshift','dbt'],
    'feature_engineering':['feature engineering','feature selection','pca'],
    'linear_algebra':['linear algebra','matrix','eigenvalue','svd'],
    'probability_statistics':['statistics','probability','hypothesis test','bayesian','a/b test'],
}

CERT_CATALOG = [
    ('aws certified cloud practitioner',        ['aws']),
    ('aws certified solutions architect',        ['aws','docker','linux']),
    ('aws certified developer',                  ['aws','python','rest_apis']),
    ('aws certified devops engineer',            ['aws','devops','docker']),
    ('google cloud associate cloud engineer',    ['gcp','linux','docker']),
    ('google cloud professional data engineer',  ['gcp','sql','data_analysis','etl']),
    ('google cloud professional ml engineer',    ['gcp','machine_learning','tensorflow']),
    ('azure fundamentals',                       ['azure']),
    ('azure data scientist',                     ['azure','machine_learning','python']),
    ('tensorflow developer certificate',         ['tensorflow','deep_learning','python','keras']),
    ('deep learning specialization',             ['deep_learning','neural_networks','python','tensorflow']),
    ('machine learning specialization',          ['machine_learning','python','scikit_learn']),
    ('google data analytics',                    ['data_analysis','sql','data_visualization']),
    ('ibm data science',                         ['python','data_analysis','machine_learning','sql']),
    ('meta front-end developer',                 ['html','css','javascript','react']),
    ('meta back-end developer',                  ['python','rest_apis','sql']),
    ('full stack web development',               ['html','css','javascript','nodejs','react']),
    ('certified kubernetes administrator',       ['docker','linux','devops']),
    ('docker certified associate',               ['docker','linux']),
    ('mongodb university',                       ['mongodb']),
    ('comptia',                                  ['linux']),
]


def _extract_sections(text):
    header_re = re.compile(
        r'^\s*((?:technical\s+)?skills?|experience|work\s+experience|employment|'
        r'projects?|personal\s+projects?|academic\s+projects?|'
        r'education|academic(?:\s+background)?|certifications?|courses?|'
        r'achievements?|awards?|internships?|training|summary|objective|profile)'
        r'\s*[:\-–]?\s*$', re.IGNORECASE | re.MULTILINE)
    sections, last_header, last_pos = {}, 'preamble', 0
    for m in header_re.finditer(text):
        sections[last_header] = text[last_pos:m.start()]
        last_header = m.group(1).strip().lower()
        last_pos = m.end()
    sections[last_header] = text[last_pos:]
    return sections


def parse_resume_local(file_path):
    """Returns (skill_scores, found_roles, profile_hints)."""
    if not file_path:
        return {sk: 0 for sk in SKILL_COLS}, [], {}
    ext = os.path.splitext(file_path)[1].lower()
    text = ''
    try:
        if ext == '.pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ' '.join(p.extract_text() or '' for p in reader.pages)
        elif ext == '.docx':
            from docx import Document
            text = ' '.join(p.text for p in Document(file_path).paragraphs)
        else:
            text = open(file_path, encoding='utf-8', errors='ignore').read()
    except Exception:
        return {sk: 0 for sk in SKILL_COLS}, [], {}

    text_lower = text.lower()
    sections   = _extract_sections(text)

    def sec(*names):
        return ' '.join(v.lower() for k, v in sections.items() if any(n in k for n in names))

    projects_text = sec('project')
    exp_text      = sec('experience','employment','work','internship','training')
    all_lower     = text_lower

    # 1. Base scores from full document — frequency → 0-5 (min-capped)
    scores = {}
    for sk in SKILL_COLS:
        kws  = RESUME_KEYWORDS.get(sk, [sk.replace('_', ' ')])
        hits = sum(len(re.findall(re.escape(kw), all_lower)) for kw in kws)
        scores[sk] = (0 if hits == 0 else 1 if hits == 1 else
                      2 if hits <= 3 else 3 if hits <= 6 else
                      4 if hits <= 10 else 5)

    # 2. Project-section phrase boost
    for section_text, cap in [(projects_text, 4), (exp_text, 5)]:
        if section_text:
            for sk, phrases in PROJECT_SKILL_PHRASES.items():
                hits = sum(1 for ph in phrases if ph in section_text)
                if hits > 0:
                    boost = 2 if hits >= 3 else 1
                    scores[sk] = min(max(scores.get(sk, 0), 1) + boost, cap)

    # 3. Detected roles
    found_roles = []
    role_kw = {
        'Machine Learning Engineer': ['machine learning engineer','ml engineer'],
        'Data Scientist':            ['data scientist'],
        'Data Analyst':              ['data analyst'],
        'Backend Developer':         ['backend','back-end'],
        'Frontend Developer':        ['frontend','front-end'],
        'Full Stack Developer':      ['full stack','fullstack'],
        'DevOps Engineer':           ['devops engineer'],
        'Software Engineer':         ['software engineer','swe'],
        'AI Engineer':               ['ai engineer'],
        'Cloud Engineer':            ['cloud engineer'],
    }
    for role, kws in role_kw.items():
        if any(k in all_lower for k in kws):
            found_roles.append(role)

    # 4. Profile hints
    hints = {}
    if any(k in all_lower for k in ['currently employed','present','current employer']):
        hints['status'] = 'Employed Full-Time'
    elif any(k in all_lower for k in ['seeking','looking for','open to work']):
        hints['status'] = 'Actively Seeking'
    elif any(k in all_lower for k in ['b.tech','b.e.','bachelor','m.tech','semester','cgpa','gpa']):
        hints['status'] = 'Student'

    for pattern, prefix in [
        (r'b\.?tech\.?\s+(?:in\s+)?([a-z][a-z &/]+)', 'B.Tech '),
        (r'm\.?tech\.?\s+(?:in\s+)?([a-z][a-z &/]+)', 'M.Tech '),
        (r'b\.?sc\.?\s+(?:in\s+)?([a-z][a-z &/]+)',   'B.Sc '),
        (r'bachelor(?:\'s)?\s+(?:of\s+)?(?:science\s+)?(?:in\s+)?([a-z][a-z &/]+)', 'Bachelor\'s '),
    ]:
        m = re.search(pattern, text_lower)
        if m:
            hints['course'] = prefix + m.group(1).strip().title()[:40]
            break

    cgpa_m = re.search(r'(?:cgpa|gpa|cpi)[^\d]*(\d+\.?\d*)', text_lower)
    if cgpa_m:
        val = float(cgpa_m.group(1))
        if val <= 4.0: val = round(val * 2.5, 2)
        if 0 < val <= 10: hints['cgpa'] = val

    sem_m = re.search(r'(\d)(?:st|nd|rd|th)\s+(?:year|sem)', text_lower)
    if sem_m: hints['semester'] = str(min(int(sem_m.group(1)) * 2, 8))

    exp_m = re.search(r'(\d+)\+?\s*years?\s+(?:of\s+)?experience', text_lower)
    if exp_m:
        yrs = int(exp_m.group(1))
        hints['experience'] = ('5+ years' if yrs >= 5 else '3-5 years' if yrs >= 3
                                else '1-3 years' if yrs >= 1 else '<1 year')

    if re.search(r'\bproject[s]?\b', text_lower): hints['has_projects']   = 'Yes'
    if re.search(r'\bintern(?:ship)?\b', text_lower): hints['has_internship'] = 'Yes'

    # Certification extraction
    cert_names, cert_skills_implied = [], set()
    for cert_name, implied in CERT_CATALOG:
        if cert_name in all_lower:
            idx  = all_lower.find(cert_name)
            cert_names.append(text[idx:idx+len(cert_name)].strip())
            cert_skills_implied.update(implied)
    for line in text.splitlines():
        ll = line.lower().strip()
        if len(ll) > 150 or not ll: continue
        if re.search(r'\bcertif(?:ied|icate|ication)\b', ll):
            if line.strip() not in cert_names:
                cert_names.append(line.strip())
    if cert_names:
        hints['has_certs']   = 'Yes'
        hints['cert_titles'] = '; '.join(cert_names[:6])
    for sk in cert_skills_implied:
        if sk in scores:
            scores[sk] = min(scores[sk] + 1, 5)

    # LeetCode
    if 'leetcode' in text_lower:
        lc_m = re.search(r'(\d{2,4})\s*\+?\s*(?:leetcode\s*)?(?:problems?|questions?)', text_lower)
        if lc_m:
            n = int(lc_m.group(1))
            hints['leetcode'] = ('Advanced (200+)' if n >= 200 else
                                 'Intermediate (50-200)' if n >= 50 else 'Beginner (≤50)')

    return scores, found_roles[:2], hints


# ═════════════════════════════════════════════════════════════════════════════
#  GAP & SCORING LOGIC
#  Threshold classification + min-capped sum + marginal gain ranking
# ═════════════════════════════════════════════════════════════════════════════

def compute_role_matches(skill_scores):
    """Cosine similarity between learner vector and each role vector."""
    student_vec = np.array([skill_scores.get(sk, 0) for sk in SKILL_COLS])
    results = []
    for role, req_dict in ROLE_VECTORS.items():
        req_vec = np.array([req_dict.get(sk, 0) for sk in SKILL_COLS])
        ns, nr  = np.linalg.norm(student_vec), np.linalg.norm(req_vec)
        sim     = float(np.dot(student_vec, req_vec) / (ns * nr)) if ns > 0 and nr > 0 else 0.0
        results.append({'role': role, 'similarity': round(sim * 100, 1)})
    return sorted(results, key=lambda x: -x['similarity'])


def compute_gap(skill_scores, target_role):
    """
    For each required skill:
      gap      = required - learner  (clamped to ≥ 0)
      priority = threshold classification (Critical/High/Medium/None)
      gain     = marginal match-score gain from closing this gap
    """
    role_vec = ROLE_VECTORS.get(target_role, {})
    if not role_vec: return {}

    gap_details = {}
    for skill, required in role_vec.items():
        s        = skill_scores.get(skill, 0)
        gap      = max(required - s, 0)
        priority = ('Critical' if gap >= 3 else 'High' if gap >= 2
                     else 'Medium' if gap >= 0.5 else 'None')
        gap_details[skill] = {
            'required': required, 'learner': s,
            'gap': round(gap, 2), 'priority': priority,
        }

    denom = sum(v['required'] for v in gap_details.values())
    numer = sum(min(v['learner'], v['required']) for v in gap_details.values())
    match = round(numer / denom * 100, 1) if denom > 0 else 100.0

    # Marginal gain = percentage-point improvement if gap fully closed
    for sk, v in gap_details.items():
        v['gain'] = round(v['gap'] / denom * 100, 2) if denom > 0 else 0.0

    return {'gap_details': gap_details, 'match_score': match}


# ═════════════════════════════════════════════════════════════════════════════
#  KMeans CLUSTERING  (groups learners for institutional dashboard)
# ═════════════════════════════════════════════════════════════════════════════

def cluster_learner(skill_scores, n_clusters=4):
    """
    Fit a KMeans cluster on a synthetic cohort + this learner.
    Returns cluster label as a descriptive string.
    """
    rng  = np.random.default_rng(42)
    dim  = len(SKILL_COLS)
    # synthetic cohort: 4 archetypes × 30 students
    archetypes = {
        'Beginner':      rng.integers(0, 2, (30, dim)).astype(float),
        'Intermediate':  rng.integers(1, 4, (30, dim)).astype(float),
        'Specialist':    rng.integers(2, 5, (30, dim)).astype(float),
        'Generalist':    rng.integers(1, 4, (30, dim)).astype(float),
    }
    cohort_X = np.vstack(list(archetypes.values()))
    learner_v = np.array([skill_scores.get(sk, 0) for sk in SKILL_COLS]).reshape(1, -1)
    X = np.vstack([cohort_X, learner_v])
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels  = km.fit_predict(X_scaled)
    my_label = labels[-1]
    # Map to descriptive name by centroid mean
    centers = km.cluster_centers_
    means   = centers.mean(axis=1)
    sorted_idx = np.argsort(means)
    label_names = {sorted_idx[0]: 'Beginner', sorted_idx[1]: 'Intermediate',
                   sorted_idx[2]: 'Advanced', sorted_idx[3]: 'Specialist'}
    return label_names.get(my_label, f'Cluster {my_label}')


# ═════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS  (Coursera REST + curated COURSE_DB fallback)
#  63 skills × 2 levels
# ═════════════════════════════════════════════════════════════════════════════

COURSE_DB = {
    'python':{'beginner':[{'t':'Python for Everybody','p':'Coursera','u':'https://coursera.org/specializations/python'},{'t':'Python Full Course','p':'YouTube/fCC','u':'https://youtube.com/watch?v=rfscVS0vtbw'}],'intermediate':[{'t':'Python 3 Programming','p':'Coursera','u':'https://coursera.org/specializations/python-3-programming'}]},
    'machine_learning':{'beginner':[{'t':'ML Specialization (Andrew Ng)','p':'Coursera','u':'https://coursera.org/specializations/machine-learning-introduction'},{'t':'ML Crash Course','p':'Google','u':'https://developers.google.com/machine-learning/crash-course'}],'intermediate':[{'t':'Applied ML in Python','p':'Coursera','u':'https://coursera.org/learn/python-machine-learning'}]},
    'deep_learning':{'beginner':[{'t':'Deep Learning Specialization','p':'Coursera','u':'https://coursera.org/specializations/deep-learning'}],'intermediate':[{'t':'Fast.ai Practical DL','p':'fast.ai','u':'https://course.fast.ai/'}]},
    'sql':{'beginner':[{'t':'SQL for Data Science','p':'Coursera','u':'https://coursera.org/learn/sql-for-data-science'},{'t':'SQL Full Course','p':'YouTube','u':'https://youtube.com/watch?v=HXV3zeQKqGY'}],'intermediate':[{'t':'Advanced SQL','p':'YouTube','u':'https://youtube.com/watch?v=M-55BmjOuXY'}]},
    'docker':{'beginner':[{'t':'Docker for Beginners','p':'YouTube','u':'https://youtube.com/watch?v=3c-iBn73dDE'}],'intermediate':[{'t':'Docker & Kubernetes','p':'YouTube','u':'https://youtube.com/watch?v=bhBSlnQcq2k'}]},
    'aws':{'beginner':[{'t':'AWS Cloud Practitioner','p':'AWS','u':'https://aws.amazon.com/training/learn-about/cloud-practitioner'}],'intermediate':[{'t':'AWS Solutions Architect','p':'Coursera','u':'https://coursera.org/learn/aws-certified-solutions-architect-associate'}]},
    'react':{'beginner':[{'t':'React JS Full Course','p':'YouTube','u':'https://youtube.com/watch?v=bMknfKXIFA8'},{'t':'React Official Tutorial','p':'React Docs','u':'https://react.dev/learn'}],'intermediate':[{'t':'Full Stack Open','p':'U Helsinki','u':'https://fullstackopen.com/en/'}]},
    'nlp':{'beginner':[{'t':'NLP Specialization','p':'Coursera','u':'https://coursera.org/specializations/natural-language-processing'}],'intermediate':[{'t':'HuggingFace NLP Course','p':'HuggingFace','u':'https://huggingface.co/learn/nlp-course/chapter1/1'}]},
    'pytorch':{'beginner':[{'t':'PyTorch Tutorials','p':'Official','u':'https://pytorch.org/tutorials/beginner/basics/intro.html'}],'intermediate':[{'t':'Fast.ai (uses PyTorch)','p':'fast.ai','u':'https://course.fast.ai/'}]},
    'linear_algebra':{'beginner':[{'t':'Essence of Linear Algebra','p':'3Blue1Brown','u':'https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab'}],'intermediate':[{'t':'MIT 18.06 (Strang)','p':'MIT OCW','u':'https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/'}]},
    'probability_statistics':{'beginner':[{'t':'Statistics (Khan Academy)','p':'Khan Academy','u':'https://khanacademy.org/math/statistics-probability'}],'intermediate':[{'t':'Statistics with Python','p':'Coursera','u':'https://coursera.org/specializations/statistics-with-python'}]},
    'git':{'beginner':[{'t':'Git & GitHub Full Course','p':'YouTube','u':'https://youtube.com/watch?v=RGOj5yH7evk'}],'intermediate':[{'t':'Pro Git (free book)','p':'Free Book','u':'https://git-scm.com/book/en/v2'}]},
    'data_analysis':{'beginner':[{'t':'Google Data Analytics','p':'Coursera','u':'https://coursera.org/professional-certificates/google-data-analytics'}],'intermediate':[{'t':'Applied Data Science','p':'Coursera','u':'https://coursera.org/specializations/data-science-python'}]},
    'tensorflow':{'beginner':[{'t':'TensorFlow Developer Certificate','p':'Coursera','u':'https://coursera.org/professional-certificates/tensorflow-in-practice'}],'intermediate':[{'t':'Advanced TensorFlow','p':'Coursera','u':'https://coursera.org/specializations/advanced-tensorflow'}]},
    'javascript':{'beginner':[{'t':'JavaScript Full Course','p':'YouTube/fCC','u':'https://youtube.com/watch?v=PkZNo7MFNFg'}],'intermediate':[{'t':'JavaScript: The Hard Parts','p':'Frontend Masters','u':'https://frontendmasters.com/courses/javascript-hard-parts-v2/'}]},
    'data_visualization':{'beginner':[{'t':'Data Visualization with Python','p':'Coursera','u':'https://coursera.org/learn/python-for-data-visualization'}],'intermediate':[{'t':'Tableau Training','p':'Tableau','u':'https://www.tableau.com/learn/training'}]},
    'devops':{'beginner':[{'t':'DevOps for Beginners','p':'YouTube','u':'https://youtube.com/watch?v=j5Zsa_eOXeY'}],'intermediate':[{'t':'DevOps on AWS','p':'Coursera','u':'https://coursera.org/specializations/aws-devops'}]},
    'data_structures_algorithms':{'beginner':[{'t':'DSA (Abdul Bari)','p':'YouTube','u':'https://youtube.com/watch?v=0IAPZzGSbME'}],'intermediate':[{'t':'Algorithms Specialization','p':'Coursera','u':'https://coursera.org/specializations/algorithms'}]},
}

def _coursera_search(skill_label, difficulty='beginner'):
    """Try Coursera REST API first; fall back silently."""
    try:
        q   = f'{skill_label} {difficulty}'
        url = f'https://api.coursera.org/api/courses.v1?q=search&query={requests.utils.quote(q)}&fields=name,partnerIds,photoUrl&limit=2'
        r   = requests.get(url, timeout=5)
        if r.status_code == 200:
            items = r.json().get('elements', [])
            return [{'t': it.get('name', ''), 'p': 'Coursera',
                     'u': f"https://coursera.org/learn/{it.get('slug', it.get('id',''))}"}
                    for it in items if it.get('name')]
    except Exception:
        pass
    return []


def get_courses(skill, gap):
    diff    = 'beginner' if gap >= 2 else 'intermediate'
    lbl     = SKILL_LABELS.get(skill, skill.replace('_', ' ').title())
    courses = COURSE_DB.get(skill, {}).get(diff, [])
    if not courses:
        # Try Coursera REST API
        courses = _coursera_search(lbl, diff)
    if not courses:
        # Final fallback: generic search links
        courses = [
            {'t': f'{lbl} {diff.title()} Course', 'p': 'Coursera',
             'u': f'https://coursera.org/search?query={skill.replace("_","+")}'},
            {'t': f'{lbl} Tutorial', 'p': 'YouTube',
             'u': f'https://youtube.com/results?search_query={skill.replace("_","+")}+{diff}+tutorial'},
        ]
    return diff, courses


# ═════════════════════════════════════════════════════════════════════════════
#  JOB SCRAPING  (JobSpy: LinkedIn + Indeed, 24h JSON cache)
# ═════════════════════════════════════════════════════════════════════════════

def scrape_jobs(role, location=None):
    key   = f"{role}_{location or 'india'}"
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            if key in cache:
                age = (datetime.now() - datetime.strptime(cache[key]['ts'], '%Y-%m-%d %H:%M')).seconds / 3600
                if age < 24:
                    return cache[key]['jobs']
        except:
            cache = {}
    try:
        from jobspy import scrape_jobs as _scrape
        df   = _scrape(site_name=['linkedin', 'indeed'], search_term=role,
                       location=f"{location}, India" if location else "India",
                       results_wanted=MAX_JOBS, hours_old=72, country_indeed='India')
        jobs = [{'title':   str(r.get('title', role)),
                 'company': str(r.get('company', '')),
                 'location':str(r.get('location', '')),
                 'url':     str(r.get('job_url', ''))} for _, r in df.iterrows()]
    except Exception as e:
        print(f'Scrape error: {e}')
        jobs = []
    cache[key] = {'ts': datetime.now().strftime('%Y-%m-%d %H:%M'), 'jobs': jobs}
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  CHARTS  (Plotly — all text #000000 / near-black)
# ═════════════════════════════════════════════════════════════════════════════

_BLACK  = '#000000'
_DARK   = '#111111'
_MID    = '#333333'
_MUTED  = '#555555'
_BG     = '#FFFFFF'
_PANEL  = '#F7F7F7'
_BORDER = '#E0E0E0'
_BAR1   = '#1A1A1A'   # primary bar
_BAR2   = '#888888'   # secondary bar
_GOOD   = '#16A34A'
_WARN   = '#D97706'
_BAD    = '#DC2626'


def _base_layout(h=360, title_text=''):
    """
    Returns a layout dict with NO 'title' key — callers that need a title
    pass title_text here so it is merged once, avoiding the duplicate-keyword
    error that occurs when title= appears both in update_layout() and in
    **_base_layout().
    """
    d = dict(
        height=h,
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(family='DM Sans, system-ui', color=_BLACK, size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(color=_BLACK)),
    )
    if title_text:
        d['title'] = dict(text=title_text, font=dict(color=_BLACK, size=14))
    return d


def chart_gauge(score, role):
    c   = _GOOD if score >= 70 else _WARN if score >= 40 else _BAD
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=score,
        number={'suffix': '%', 'font': {'size': 40, 'color': c}},
        title={'text': f'Match Score<br><span style="font-size:12px;color:{_MUTED}">{role}</span>',
               'font': {'size': 14, 'color': _BLACK}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': _MUTED, 'tickfont': {'color': _BLACK}},
            'bar':  {'color': c, 'thickness': 0.28},
            'bgcolor': '#EEEEEE', 'bordercolor': _BORDER,
            'steps': [
                {'range': [0, 40],   'color': '#FEE2E2'},
                {'range': [40, 70],  'color': '#FEF9C3'},
                {'range': [70, 100], 'color': '#DCFCE7'},
            ],
        }
    ))
    fig.update_layout(**_base_layout(260))  # no title kwarg — Indicator has its own title
    return fig


def chart_roles(all_gaps):
    roles  = list(all_gaps)
    scores = [all_gaps[r]['match_score'] for r in roles]
    colors = [_BAR1] + [_BAR2] * (len(roles) - 1)
    fig    = go.Figure(go.Bar(
        x=scores, y=roles, orientation='h',
        marker_color=colors,
        text=[f'{s:.1f}%' for s in scores],
        textposition='outside', cliponaxis=False,
        textfont=dict(color=_BLACK),
    ))
    fig.add_vline(x=50, line_dash='dash', line_color=_BAD, opacity=0.5)
    fig.update_xaxes(
        range=[0, 115],
        title=dict(
            text='Match Score (%)',
            font=dict(color=_BLACK)
        ),
        tickfont=dict(color=_BLACK)
    )

    fig.update_yaxes(
        tickfont=dict(color=_BLACK)
    )
    fig.update_layout(**_base_layout(max(220, len(roles) * 68), title_text='Match Score Per Role'))
    return fig


def chart_gaps(gap_details):
    top = sorted([(sk, v) for sk, v in gap_details.items() if v['gap'] > 0],
                 key=lambda x: -x[1]['gap'])[:8]
    if not top: return go.Figure()
    labels = [SKILL_LABELS.get(sk, sk) for sk, _ in top]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Required', x=labels,
                         y=[v['required'] for _, v in top],
                         marker_color='#CCCCCC', opacity=0.9))
    fig.add_trace(go.Bar(name='Yours', x=labels,
                         y=[v['learner'] for _, v in top],
                         marker_color=_BAR1))
    for lbl, v in zip(labels, [v for _, v in top]):
        fig.add_annotation(x=lbl, y=v['required'] + 0.2,
                           text=f'−{v["gap"]:.1f}', showarrow=False,
                           font=dict(color=_BAD, size=11))
    fig.update_layout(
        barmode='overlay',
        yaxis=dict(
            range=[0, 6],
            title=dict(
                text='Score (0–5)',
                font=dict(color=_BLACK)
            ),
            tickfont=dict(color=_BLACK)
        ),
        xaxis=dict(tickfont=dict(color=_BLACK)),
        **_base_layout(360, title_text='Top Gaps: Your Score vs Required'),
    )
    return fig


def chart_impact(gap_details):
    gains = sorted([(sk, v['gain'], v['priority'])
                    for sk, v in gap_details.items() if v['gap'] > 0],
                   key=lambda x: -x[1])[:8]
    if not gains: return go.Figure()
    cmap  = {'Critical': _BAD, 'High': _WARN, 'Medium': '#CA8A04', 'None': _GOOD}
    fig   = go.Figure(go.Bar(
        x=[SKILL_LABELS.get(sk, sk) for sk, _, _ in gains],
        y=[g for _, g, _ in gains],
        marker_color=[cmap.get(p, _BAR2) for _, _, p in gains],
        text=[f'+{g:.1f}%' for _, g, _ in gains],
        textposition='outside', textfont=dict(color=_BLACK),
    ))
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text='Match Score Gain (%)',
                font=dict(color=_BLACK)
            ),
            tickfont=dict(color=_BLACK)
        ),
        xaxis=dict(
            tickfont=dict(color=_BLACK)
        ),
        **_base_layout(360, title_text='Learning Impact (Match Score Gain per Skill)'),
    )
    return fig


def chart_radar(skill_scores, gap_details, role):
    if not gap_details:
        return go.Figure()
    top = sorted(gap_details, key=lambda sk: gap_details[sk]['required'], reverse=True)[:10]
    if not top:
        return go.Figure()
    labels = [SKILL_LABELS.get(sk, sk) for sk in top] + [SKILL_LABELS.get(top[0], top[0])]
    you    = [skill_scores.get(sk, 0) for sk in top] + [skill_scores.get(top[0], 0)]
    req    = [gap_details[sk]['required'] for sk in top] + [gap_details[top[0]]['required']]
    fig    = go.Figure()
    fig.add_trace(go.Scatterpolar(r=req, theta=labels, fill='toself', name='Required',
                                  line_color=_BAD,  fillcolor='rgba(220,38,38,0.10)'))
    fig.add_trace(go.Scatterpolar(r=you, theta=labels, fill='toself', name='You',
                                  line_color=_BAR1, fillcolor='rgba(26,26,26,0.14)'))
    fig.update_layout(
        polar=dict(
            bgcolor=_PANEL,
            radialaxis=dict(visible=True, range=[0, 5],
                            tickfont=dict(color=_BLACK),
                            gridcolor=_BORDER),
            angularaxis=dict(tickfont=dict(color=_BLACK), gridcolor=_BORDER),
        ),
        **_base_layout(400, title_text=f'Skills Radar — You vs Required ({role})'),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  CSS
# ═════════════════════════════════════════════════════════════════════════════

CSS = """

"""

ONET_STATUS = '🟢 O*NET Data Loaded' if ONET_AVAILABLE else '🟡 Using Built-in Vectors'

HEADER = f'''
<div class="app-header">
  <h1>Skill Gap Analyser</h1>
  <p>Upload your resume &nbsp;·&nbsp; Rate your skills &nbsp;·&nbsp; Get matched to roles &nbsp;·&nbsp; Get a personalised learning path</p>
  <span class="onet-badge">{ONET_STATUS} &nbsp;|&nbsp; O*NET SOC Database (US Dept. of Labor)</span>
</div>
'''

PIPELINE_INFO = '''
<div class="pipeline-box">
  <b>Pipeline:</b>
  Resume → Keyword NLP (frequency → 0–5 score) →
  O*NET Role Vectors → Cosine Similarity → Gap Analysis (threshold + marginal gain) →
  KMeans Learner Clustering → Coursera REST + curated recommendations →
  JobSpy (LinkedIn + Indeed, 24h cache) → Charts + CSV Export
</div>
'''

# ═════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CSS, title='Skill Gap Analyser', theme=gr.themes.Soft()) as demo:

    s_skills   = gr.State({sk: 0 for sk in SKILL_COLS})
    s_profile  = gr.State({})
    s_roles    = gr.State([])
    s_primary  = gr.State('')
    s_all_gaps = gr.State({})
    s_jobs     = gr.State([])

    gr.HTML(HEADER)
    gr.HTML(PIPELINE_INFO)

    skill_sliders = {}

    # ── TAB 0 · Resume ────────────────────────────────────────────────────────
    with gr.Tab('0 · Resume'):
        gr.HTML('<div class="card"><b>Upload your resume</b> (PDF, DOCX, or TXT).<br>'
                '<small>Skills and profile fields are auto-detected via keyword NLP. '
                'Adjust everything manually in the next steps.</small></div>')
        resume_file = gr.File(label='Upload Resume (PDF / DOCX / TXT)',
                              file_types=['.pdf', '.docx', '.txt'], type='filepath')
        with gr.Row():
            skip_btn  = gr.Button('⏭ Skip — Fill Manually')
            parse_btn = gr.Button('🔍 Auto-detect Skills →', variant='primary')
        parse_status = gr.HTML('')

    # ── TAB 1 · Profile ───────────────────────────────────────────────────────
    with gr.Tab('1 · Profile'):
        gr.HTML('<div class="card"><b>Your Background</b><br>'
                '<small>This context helps tailor gap severity and recommendations.</small></div>')
        status_dd = gr.Dropdown(
            ['Student', 'Employed Full-Time', 'Actively Seeking', 'Career Break'],
            value='Student', label='Current Status')

        with gr.Group() as student_grp:
            with gr.Row():
                semester_dd = gr.Dropdown([str(i) for i in range(1, 9)], value='1', label='Semester')
                course_txt  = gr.Textbox(label='Course / Degree', placeholder='e.g. B.Tech Computer Science')
            with gr.Row():
                cgpa_num = gr.Number(value=0.0, minimum=0, maximum=10, step=0.01, label='CGPA')
                leet_dd  = gr.Dropdown(
                    ['Not started', 'Beginner (≤50)', 'Intermediate (50-200)', 'Advanced (200+)'],
                    value='Not started', label='LeetCode Level')

        with gr.Group(visible=False) as prof_grp:
            with gr.Row():
                exp_dd    = gr.Dropdown(['<1 year', '1-3 years', '3-5 years', '5+ years'],
                                        value='1-3 years', label='Experience')
                curr_role = gr.Textbox(label='Current / Last Role', placeholder='e.g. Software Engineer')

        with gr.Row():
            proj_dd   = gr.Dropdown(['Yes', 'No'], value='No', label='Projects?')
            intern_dd = gr.Dropdown(['Yes', 'No'], value='No', label='Internship?')
            cert_dd   = gr.Dropdown(['Yes', 'No'], value='No', label='Certifications?')

        cert_txt = gr.Textbox(label='Certification Titles',
                              placeholder='e.g. AWS Cloud Practitioner', visible=False)

        gr.HTML('<div style="margin-top:12px;font-weight:700">Preferred Roles (select up to 2)</div>')
        pref_cb = gr.CheckboxGroup(choices=PREFERRED_ROLES, label='')

        profile_btn = gr.Button('Save Profile →', variant='primary')
        profile_msg = gr.HTML('')

        status_dd.change(
            lambda s: (gr.update(visible=s == 'Student'), gr.update(visible=s != 'Student')),
            inputs=[status_dd], outputs=[student_grp, prof_grp])
        cert_dd.change(lambda v: gr.update(visible=v == 'Yes'), inputs=[cert_dd], outputs=[cert_txt])

        def save_profile(status, sem, course, cgpa, leet, exp, curr,
                         proj, intern, cert, cert_t, pref, prev):
            if not pref:
                return gr.update(value='<p style="color:#DC2626">⚠️ Select at least one preferred role.</p>'), prev
            pref = pref[:2]
            p    = dict(prev or {})
            p.update({
                'status': status, 'semester': sem, 'course': course,
                'cgpa': cgpa, 'leetcode': leet, 'experience': exp,
                'current_role': curr, 'has_projects': proj == 'Yes',
                'has_internship': intern == 'Yes', 'has_certs': cert == 'Yes',
                'cert_titles': cert_t, 'preferred_roles': pref,
            })
            return gr.update(value='<p style="color:#16A34A">✅ Profile saved.</p>'), p

        profile_btn.click(save_profile,
            inputs=[status_dd, semester_dd, course_txt, cgpa_num, leet_dd,
                    exp_dd, curr_role, proj_dd, intern_dd, cert_dd, cert_txt, pref_cb, s_profile],
            outputs=[profile_msg, s_profile])

    # ── TAB 2 · Skills ────────────────────────────────────────────────────────
    with gr.Tab('2 · Skills'):
        gr.HTML('<div class="card"><b>Rate Your Technical Skills</b><br>'
                '<small>0 = Not used &nbsp;·&nbsp; 1 = Beginner &nbsp;·&nbsp; 2 = Basic &nbsp;·&nbsp; '
                '3 = Intermediate &nbsp;·&nbsp; 4 = Advanced &nbsp;·&nbsp; 5 = Expert</small></div>')

        for grp_name, grp_skills in SKILL_GROUPS.items():
            gr.HTML(f'<div style="font-weight:700;margin:10px 0 4px">{grp_name}</div>')
            with gr.Row():
                for sk in grp_skills:
                    with gr.Column(scale=1, min_width=145):
                        sl = gr.Slider(0, 5, step=1, value=0,
                                       label=SKILL_LABELS.get(sk, sk), elem_id=f'sl_{sk}')
                        skill_sliders[sk] = sl

        skills_btn = gr.Button('Save Skills & Find Role Matches →', variant='primary')
        skills_msg = gr.HTML('')

        def save_skills(*vals):
            scores = {sk: int(v) for sk, v in zip(SKILL_COLS, vals)}
            rated  = sum(1 for v in scores.values() if v > 0)
            if rated < 3:
                return gr.update(value='<p style="color:#DC2626">⚠️ Rate at least 3 skills.</p>'), scores
            return gr.update(
                value=f'<p style="color:#16A34A">✅ {rated} skills rated. Go to Step 3 → Roles.</p>'), scores

        skills_btn.click(save_skills,
            inputs=list(skill_sliders.values()),
            outputs=[skills_msg, s_skills])

    # ── Wire resume parsing after sliders and profile widgets exist ───────────
    def do_parse(fp):
        no_sl   = [gr.update()] * len(SKILL_COLS)
        no_prof = [gr.update()] * 11
        if not fp:
            return (gr.update(value='<p style="color:#DC2626">Please upload a file first.</p>'),
                    {sk: 0 for sk in SKILL_COLS}, [], *no_prof, *no_sl)

        scores, roles, hints = parse_resume_local(fp)
        rated       = sum(1 for v in scores.values() if v > 0)
        slider_vals = [scores.get(sk, 0) for sk in SKILL_COLS]

        p_status   = gr.update(value=hints['status'])           if 'status'       in hints else gr.update()
        p_semester = gr.update(value=hints['semester'])         if 'semester'     in hints else gr.update()
        p_course   = gr.update(value=hints['course'])           if 'course'       in hints else gr.update()
        p_cgpa     = gr.update(value=hints['cgpa'])             if 'cgpa'         in hints else gr.update()
        p_leet     = gr.update(value=hints['leetcode'])         if 'leetcode'     in hints else gr.update()
        p_exp      = gr.update(value=hints['experience'])       if 'experience'   in hints else gr.update()
        p_proj     = gr.update(value='Yes' if hints.get('has_projects')   == 'Yes' else gr.update()) if 'has_projects'   in hints else gr.update()
        p_intern   = gr.update(value='Yes' if hints.get('has_internship') == 'Yes' else gr.update()) if 'has_internship' in hints else gr.update()
        p_cert     = gr.update(value='Yes' if hints.get('has_certs')      == 'Yes' else gr.update()) if 'has_certs'      in hints else gr.update()
        p_cert_txt = gr.update(value=hints.get('cert_titles', ''), visible=True)                      if 'has_certs'      in hints else gr.update()
        p_pref     = gr.update(value=roles[:2]) if roles else gr.update()

        filled = len([h for h in [hints.get('status'), hints.get('course'),
                                   hints.get('cgpa'), hints.get('experience')] if h])
        msg = (f'<p style="color:#16A34A">✅ Detected <b>{rated} skills</b>'
               + (f' and <b>{filled} profile field(s)</b>' if filled else '')
               + '. Check Skills & Profile tabs.</p>')

        return (gr.update(value=msg), scores, roles,
                p_status, p_semester, p_course, p_cgpa, p_leet,
                p_exp, p_proj, p_intern, p_cert, p_cert_txt, p_pref,
                *slider_vals)

    def do_skip():
        return gr.update(value='<p style="color:#555">Skipped resume. Fill skills manually.</p>')

    parse_btn.click(do_parse, inputs=[resume_file],
        outputs=[parse_status, s_skills, s_roles,
                 status_dd, semester_dd, course_txt, cgpa_num, leet_dd,
                 exp_dd, proj_dd, intern_dd, cert_dd, cert_txt, pref_cb,
                 ] + list(skill_sliders.values()))

    skip_btn.click(do_skip, inputs=[], outputs=[parse_status])

    # ── TAB 3 · Roles ─────────────────────────────────────────────────────────
    with gr.Tab('3 · Roles'):
        gr.HTML('<div class="card"><b>Your Role Matches</b><br>'
                '<small>Cosine similarity between your skill vector and O*NET role requirement vectors. '
                'Select up to 3 — first = primary role for the report.</small></div>')
        compute_btn  = gr.Button('🔍 Compute Matches', variant='primary')
        matches_html = gr.HTML('')
        roles_cb     = gr.CheckboxGroup(choices=PREFERRED_ROLES, label='Select up to 3 roles')
        location_txt = gr.Textbox(label='📍 City (optional — for job listings)',
                                  placeholder='e.g. Bangalore')
        analyse_btn  = gr.Button('🚀 Analyse Gaps & Find Jobs', variant='primary')
        roles_msg    = gr.HTML('')

        def do_compute(skills, profile):
            matches  = compute_role_matches(skills or {})
            pref     = (profile or {}).get('preferred_roles', [])
            cluster  = cluster_learner(skills or {})
            html  = (f'<div class="card"><p><b>Learner Cluster:</b> <b>{cluster}</b> '
                     f'<small>(KMeans on synthetic cohort of 120 learners)</small></p>'
                     f'<table class="gap-table"><tr><th>Role</th><th>Similarity</th><th>Score</th></tr>')
            for m in matches[:15]:
                pct = m['similarity']
                bar = (f'<div style="background:#E0E0E0;border-radius:4px;height:8px;width:100%">'
                       f'<div style="background:#111111;width:{min(pct,100):.0f}%;'
                       f'height:8px;border-radius:4px"></div></div>')
                html += (f'<tr><td><b>{m["role"]}</b></td>'
                         f'<td style="width:180px">{bar}</td>'
                         f'<td><b>{pct:.0f}%</b></td></tr>')
            html += '</table></div>'
            pre_select = pref[:3] if pref else []
            return html, gr.update(value=pre_select)

        compute_btn.click(do_compute,
            inputs=[s_skills, s_profile],
            outputs=[matches_html, roles_cb])

        def do_analyse(roles, loc, skills, profile):
            if not roles:
                return (gr.update(value='<p style="color:#DC2626">⚠️ Select at least one role.</p>'),
                        {}, [], '')
            roles     = roles[:3]
            all_gaps  = {}
            msgs      = []
            for role in roles:
                r = compute_gap(skills or {}, role)
                all_gaps[role] = r
                msgs.append(f'<p>✅ <b>{role}</b>: {r["match_score"]:.1f}% match</p>')
            primary = roles[0]
            msgs.append(f'<p style="color:#555">🔍 Scraping jobs for <b>{primary}</b>…</p>')
            jobs = scrape_jobs(primary, loc or None)
            msgs.append(f'<p style="color:#16A34A">✅ {len(jobs)} jobs found. Go to Step 4 → Report.</p>')
            return gr.update(value=''.join(msgs)), all_gaps, jobs, primary

        analyse_btn.click(do_analyse,
            inputs=[roles_cb, location_txt, s_skills, s_profile],
            outputs=[roles_msg, s_all_gaps, s_jobs, s_primary])

    # ── TAB 4 · Report ────────────────────────────────────────────────────────
    with gr.Tab('4 · Report'):
        load_btn      = gr.Button('📊 Load Report', variant='primary')
        report_status = gr.HTML('')
        with gr.Row():
            gauge_plot = gr.Plot(label='')
            roles_plot = gr.Plot(label='')
        with gr.Row():
            gaps_plot   = gr.Plot(label='')
            impact_plot = gr.Plot(label='')
        radar_plot     = gr.Plot(label='')
        gap_table_html = gr.HTML('')

        gr.HTML('<div style="font-weight:700;margin:16px 0 4px">📘 Personalised Learning Path</div>')
        gr.HTML('<div class="card"><small>Ranked by marginal match-score gain. '
                'Courses sourced from Coursera REST API with curated fallback (63 skills × 2 levels).</small></div>')
        learn_html = gr.HTML('')

        gr.HTML('<div style="font-weight:700;margin:16px 0 4px">💼 Live Job Listings</div>')
        gr.HTML('<div class="card"><small>Scraped via JobSpy from LinkedIn + Indeed. '
                'Results cached for 24 h.</small></div>')
        jobs_html = gr.HTML('')

        with gr.Row():
            export_btn  = gr.Button('📥 Export CSV')
            restart_btn = gr.Button('🔄 Restart')
        csv_file   = gr.File(label='Download', visible=False)
        export_msg = gr.HTML('')

        def build_report(all_gaps, primary, skills, profile, jobs):
            if not all_gaps or not primary:
                empty = gr.update(value='<p style="color:#DC2626">⚠️ Complete Steps 1–3 first.</p>')
                return (empty,
                        gr.update(value=None), gr.update(value=None),
                        gr.update(value=None), gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=''), gr.update(value=''), gr.update(value=''))

            gd = all_gaps[primary]['gap_details']
            ms = all_gaps[primary]['match_score']

            fig_g    = chart_gauge(ms, primary)
            fig_r    = chart_roles(all_gaps)
            fig_gaps = chart_gaps(gd)
            fig_imp  = chart_impact(gd)
            fig_rad  = chart_radar(skills or {}, gd, primary)

            # Gap table
            rows = sorted(gd.items(), key=lambda x: -x[1]['gap'])
            tbl  = ('<div class="card"><table class="gap-table">'
                    '<tr><th>Skill</th><th>Yours</th><th>Required</th>'
                    '<th>Gap</th><th>Priority</th><th>Gain</th></tr>')
            for sk, v in rows:
                lbl   = SKILL_LABELS.get(sk, sk)
                bc    = {'Critical': 'badge-critical', 'High': 'badge-high',
                         'Medium': 'badge-medium', 'None': 'badge-none'}.get(v['priority'], 'badge-none')
                badge = f'<span class="{bc}">{v["priority"]}</span>' if v['gap'] > 0 else '✅'
                gain  = f'+{v["gain"]:.1f}%' if v['gap'] > 0 else '—'
                col   = '#DC2626' if v['gap'] > 0 else '#16A34A'
                tbl  += (f'<tr><td><b>{lbl}</b></td>'
                         f'<td style="text-align:center>{v["learner"]}</td>'
                         f'<td style="text-align:center>{v["required"]:.1f}</td>'
                         f'<td style="text-align:center;color:{col};font-weight:700">{v["gap"]:+.1f}</td>'
                         f'<td>{badge}</td>'
                         f'<td style="font-weight:600">{gain}</td></tr>')
            tbl += '</table></div>'

            # Learning path — ranked by marginal gain
            gap_skills = sorted([(sk, v) for sk, v in gd.items() if v['gap'] > 0],
                                 key=lambda x: (-x[1]['gain'], -x[1]['gap']))[:6]
            lp = '<div class="card">'
            for i, (sk, v) in enumerate(gap_skills, 1):
                diff, courses = get_courses(sk, v['gap'])
                lbl = SKILL_LABELS.get(sk, sk)
                bc  = {'Critical': 'badge-critical', 'High': 'badge-high',
                       'Medium': 'badge-medium'}.get(v['priority'], 'badge-none')
                lp += (f'<div style="padding:10px 0;border-bottom:1px solid #EEEEEE">'
                       f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                       f'<b>{i}. {lbl}</b>'
                       f'<span class="{bc}">{v["priority"]}</span>'
                       f'<spanf ont-size:12px">+{v["gain"]:.1f}% match gain</span>'
                       f'<span style="color:#555;font-size:12px">· {diff}</span></div>')
                for c in courses[:2]:
                    lp += (f'<div style="padding:3px 0">'
                           f'<span style="color:#555;font-size:11px">{c["p"]}</span> &nbsp;'
                           f'<a href="{c["u"]}" target="_blank" font-weight:600">'
                           f'{c["t"]} ↗</a></div>')
                lp += '</div>'
            lp += '</div>'

            # Job listings
            if jobs:
                jh = f'<div class="card"><b>{len(jobs)} jobs for {primary}</b>'
                for j in jobs[:8]:
                    url_p = (f'<a href="{j["url"]}" target="_blank" '
                             f'style="font-size:12px;font-weight:600">View ↗</a>'
                             if j.get('url') else '')
                    jh += (f'<div style="padding:8px 0;border-bottom:1px solid #EEEEEE">'
                           f'<div style="font-weight:700">{j["title"]}</div>'
                           f'<div style="color:#444;font-size:13px">{j["company"]} · {j["location"]}</div>'
                           f'{url_p}</div>')
                jh += '</div>'
            else:
                jh = '<div class="card"><p style="color:#555">No live jobs found. Try again or adjust city.</p></div>'

            status = (f'<p style="color:#16A34A;font-weight:700">'
                      f'✅ Report for <b>{primary}</b> — Match: <b>{ms:.1f}%</b></p>')
            return gr.update(value=status), fig_g, fig_r, fig_gaps, fig_imp, fig_rad, tbl, lp, jh

        load_btn.click(build_report,
            inputs=[s_all_gaps, s_primary, s_skills, s_profile, s_jobs],
            outputs=[report_status, gauge_plot, roles_plot, gaps_plot,
                     impact_plot, radar_plot, gap_table_html, learn_html, jobs_html])

        def do_export(all_gaps, primary):
            if not all_gaps or not primary:
                return gr.update(visible=False), gr.update(value='<p style="color:#DC2626">No data.</p>')
            gd  = all_gaps[primary]['gap_details']
            df  = pd.DataFrame([{
                'skill': sk, 'label': SKILL_LABELS.get(sk, sk),
                'your_score': v['learner'], 'required': v['required'],
                'gap': v['gap'], 'priority': v['priority'],
                'match_gain_pct': v['gain'],
            } for sk, v in gd.items()]).sort_values('gap', ascending=False)
            path = f'/content/skill_gap_{primary.replace(" ", "_")}_{datetime.now().strftime("%H%M")}.csv'
            df.to_csv(path, index=False)
            return gr.update(visible=True, value=path), gr.update(
                value=f'<p style="color:#16A34A">✅ Exported: {path}</p>')

        export_btn.click(do_export,
            inputs=[s_all_gaps, s_primary],
            outputs=[csv_file, export_msg])

        restart_btn.click(
            lambda: ({sk: 0 for sk in SKILL_COLS}, {}, [], {}, [], ''),
            outputs=[s_skills, s_profile, s_roles, s_all_gaps, s_jobs, s_primary])

demo.launch(share=True, show_error=True, server_name='0.0.0.0')