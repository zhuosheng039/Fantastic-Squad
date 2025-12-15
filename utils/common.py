from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Position list
DF = ['CB', 'LB', 'RB']
MF = ['CM', 'CDM', 'LM', 'RM', 'CAM']
FW = ['ST', 'LW', 'RW']
POS = ['DF', 'MF', 'FW', 'GK']

ATTRS = ['OVR', 'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
ATTRSEX = ATTRS + ['Age', 'Height', 'Weight']
STATS = ['Acceleration', 'Sprint Speed', 'Positioning', 'Finishing',
       'Shot Power', 'Long Shots', 'Volleys', 'Penalties', 'Vision',
       'Crossing', 'Free Kick Accuracy', 'Short Passing', 'Long Passing',
       'Curve', 'Dribbling', 'Agility', 'Balance', 'Reactions', 'Ball Control',
       'Composure', 'Interceptions', 'Heading Accuracy', 'Def Awareness',
       'Standing Tackle', 'Sliding Tackle', 'Jumping', 'Stamina', 'Strength',
       'Aggression',  'Weak foot', 'Skill moves', 'Age']

POSITION_SLOTS = [
    # FW
    ('LW', 'FW'),
    ('ST', 'FW'),
    ('RW', 'FW'),

    # MF
    ('CM', 'MF'),
    ('CM', 'MF'),
    ('CM', 'MF'),

    # DF
    ('LB', 'DF'),
    ('CB', 'DF'),
    ('CB', 'DF'),
    ('RB', 'DF'),

    # GK
    ('GK', 'GK'),
]

FORMATION_POSITIONS = {
    'GK':  [(0.5, 0.1)],

    'LB':  [(0.2, 0.25)],
    'CB': [(0.4, 0.25), (0.6, 0.25)],
    'RB':  [(0.8, 0.25)],  
           
    'CM':  [(0.3, 0.5), (0.5, 0.55), (0.7, 0.5)],

    'LW':  [(0.2, 0.8)],
    'ST':  [(0.5, 0.85)],
    'RW':  [(0.8, 0.8)],
}

POS_COLORS = {
    'GK': '#FFD700',
    'DF': '#1E90FF',
    'MF': '#3CB371',
    'FW': '#DC143C'
}
