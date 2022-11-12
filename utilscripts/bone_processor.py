from dme_loader.jenkins import oaat

HIERARCHY = {
    "AXLEBACK": "",
    "AXLEFRONT": "",
    "BARREL": "PITCH",
    "BARREL1": "PITCH",
    "BARREL2": "PITCH",
    "BARREL3": "PITCH",
    "BARREL4": "PITCH",
    "BASE": "",
    "BASEOFFSET": "",
    "BUMPER": "",
    "B_ANKLE": "",
    "B_FOOT": "",
    "B_FOOT_INSIDETOES": "",
    "B_FOOT_INSIDETOES_END": "B_FOOT_INSIDETOES",
    "B_FOOT_OUTSIDETOES": "",
    "B_FOOT_OUTSIDETOES_END": "B_FOOT_OUTSIDETOES",
    "B_FOOT_SPIKES": "",
    "B_FOOT_SPIKES_END": "B_FOOT_SPIKES",
    "B_LEG": "",
    "B_LEG_PISTON": "",
    "CAMERA": "",
    "CAMERA1_SOCKET": "",
    "CAMERARECOIL": "",
    "CANNON_HOOD": "",
    "CANNON_HOOD_END": "CANNON_HOOD",
    "COG": "WORLDROOT",
    "C_FRONTGEAR1": "COG",
    "C_FRONTGEAR2": "C_FRONTGEAR1",
    "C_FRONTGEARDOOR1": "COG",
    "C_FRONTGEARDOOR2": "COG",
    "DEPLOYFRONT1": "COG",
    "DEPLOYFRONT2": "DEPLOYFRONT1",
    "DEPLOYFRONT3": "DEPLOYFRONT2",
    "DEPLOYFRONTCOVER": "COG",
    "DISPLAY": "",
    "ELEVATOR": "",
    "END_L_BACK_CANNISTER01": "",
    "END_L_BACK_CANNISTER02": "",
    "END_L_TOP_BOX01": "",
    "END_L_TOP_BOX02": "",
    "END_L_TOP_BOX03": "",
    "END_L_TOP_BOX04": "",
    "END_L_TOP_CANNISTER01": "",
    "END_L_TOP_CANNISTER02": "",
    "END_L_TOP_CANNISTER03": "",
    "END_R_BACK_CANNISTER01": "",
    "END_R_BACK_CANNISTER02": "",
    "END_R_TOP_BOX01": "",
    "END_R_TOP_BOX02": "",
    "END_R_TOP_BOX03": "",
    "END_R_TOP_BOX04": "",
    "END_R_TOP_CANNISTER01": "",
    "END_R_TOP_CANNISTER02": "",
    "END_R_TOP_CANNISTER03": "",
    "FORWARDFLAP": "COG",
    "FRONTAXLE": "COG",
    "FRONTCHASSIS": "",
    "FRONTGEAR1": "COG",
    "FRONTGEAR2": "FRONTGEAR1",
    "FRONTGEARA": "COG",
    "FRONTGEARB": "FRONTGEARA",
    "FRONTGEARC": "FRONTGEARB",
    "FRONTGEARDOOR": "COG",
    "FRONTGEARSUPPORT": "COG",
    "FRONTSUPPORT1": "COG",
    "FRONTSUPPORT2": "FRONTSUPPORT1",
    "FRONTSUPPORT3": "FRONTSUPPORT2",
    "FRONTSUPPORT4": "FRONTSUPPORT3",
    "FRONTSUPPORT5": "FRONTSUPPORT4",
    "F_ANKLE": "",
    "F_FOOT": "",
    "F_FOOT_INSIDETOES": "",
    "F_FOOT_INSIDETOES_END": "F_FOOT_INSIDETOES",
    "F_FOOT_OUTSIDETOES": "",
    "F_FOOT_OUTSIDETOES_END": "F_FOOT_OUTSIDETOES",
    "F_FOOT_SPIKES": "",
    "F_FOOT_SPIKES_END": "F_FOOT_SPIKES",
    "F_LEG": "",
    "F_LEG_PISTON": "",
    "GEARBACK": "",
    "GEARFRONT": "",
    "HANDLEBARS": "",
    "HARDPOINT2_SOCKET": "",
    "HEAD": "NECK",
    "HEAD_END": "HEAD",
    "HYDROLICFRONTBOTTOM": "",
    "HYDROLICFRONTTOP": "",
    "HYDROLICREARBOTTOM": "",
    "HYDROLICREARTOP": "",
    "LOWERVENTS": "COG",
    "L_ANKLE": "L_KNEE",
    "L_BACKPLATE": "",
    "L_BACKSHOCKA": "",
    "L_BACKSHOCKA_TOP": "",
    "L_BACKSHOCKB": "",
    "L_BACKSHOCKB_TOP": "",
    "L_BACK_CANNISTER01": "",
    "L_BACK_CANNISTER02": "",
    "L_BALL": "L_ANKLE",
    "L_BARREL": "",
    "L_BARRELFRONT_EXTEND": "",
    "L_BARRELSIDELAYER1": "",
    "L_BARRELSIDELAYER1_END": "L_BARRELSIDELAYER1",
    "L_BARRELSIDELAYER2": "",
    "L_BARRELSIDE_EXTEND": "",
    "L_BOTTOMCANISTER": "",
    "L_BOTTOMHOUSING": "",
    "L_BOTTOMTUBE": "",
    "L_BOTTOMVENTS": "COG",
    "L_CLAVICLE": "SPINEUPPER",
    "L_CYLINDERDN": "",
    "L_CYLINDEREXTEND_END": "L_CYLINDEREXTEND",
    "L_CYLINDEREXTEND_END1": "",
    "L_CYLINDERUP": "",
    "L_DEPLOY_MECANISM": "",
    "L_DEPLOY_MECANISM_END": "L_DEPLOY_MECANISM",
    "L_ELBOW": "L_SHOULDERROLL",
    "L_FAN": "L_THRUSTER",
    "L_FLAP": "",
    "L_FLAPPER": "COG",
    "L_FOOT": "",
    "L_FOOT_INSIDETOES": "",
    "L_FOOT_INSIDETOES_END": "L_FOOT_INSIDETOES",
    "L_FOOT_OUTSIDETOES": "",
    "L_FOOT_OUTSIDETOES_END": "L_FOOT_OUTSIDETOES",
    "L_FOOT_SPIKES": "",
    "L_FOOT_SPIKES_END": "L_FOOT_SPIKES",
    "L_FOREARM": "L_ELBOW",
    "L_FRONTAXLE": "STEERING",
    "L_FRONTDOOR": "COG",
    "L_FRONTFLAP": "L_HIP",
    "L_FRONTGEAR1": "COG",
    "L_FRONTGEAR2": "L_FRONTGEAR1",
    "L_FRONTGEAR3": "L_FRONTGEAR2",
    "L_FRONTGEAR4": "L_FRONTGEAR3",
    "L_FRONTGEAR5": "COG",
    "L_FRONTGEARDOOR1": "COG",
    "L_FRONTGEARDOOR2": "COG",
    "L_FRONTPLATE": "",
    "L_FRONTSHOCKA": "",
    "L_FRONTSHOCKA_TOP": "",
    "L_FRONTSHOCKB": "",
    "L_FRONTSHOCKB_TOP": "",
    "L_FRONTSHOCKLOWER": "L_STEERINGPIVOT",
    "L_FRONTSHOCKUPPER": "COG",
    "L_FRONTSUPPORT": "L_FRONTAXLE",
    "L_FRONTSUPPORT1": "",
    "L_FRONTSUPPORT2": "",
    "L_FRONTSUPPORT3": "",
    "L_FRONTSUPPORT4": "",
    "L_FRONTSUSPENSION": "COG",
    "L_FRONTWHEEL": "FRONTAXLE",
    "L_FRONTWHEEL_DUMMY": "L_FRONTSUPPORT",
    "L_GEARA": "COG",
    "L_GEARB": "L_GEARA",
    "L_GEARC": "L_GEARB",
    "L_GEARDOOR": "COG",
    "L_GEARSUPPORTA": "L_GEARA",
    "L_GEARSUPPORTB": "COG",
    "L_HATCH_BOTTOM": "",
    "L_HATCH_BOTTOM_END": "L_HATCH_BOTTOM",
    "L_HATCH_TOP": "",
    "L_HATCH_TOP_END": "L_HATCH_TOP",
    "L_HIP": "PELVIS",
    "L_INDEXA": "L_WRIST",
    "L_INDEXB": "L_INDEXA",
    "L_INDEXC": "L_INDEXB",
    "L_INDEX_END": "L_INDEXC",
    "L_INNERFLAP": "L_THRUSTER",
    "L_INNERTHRUSTER": "L_THRUSTER",
    "L_KNEE": "L_HIP",
    "L_KNOB_ROTATE": "",
    "L_KNOB_TRANSLATE": "",
    "L_LEG": "",
    "L_LEG_PISTON": "",
    "L_LOWERSUSPENSION": "",
    "L_MIDAXLE": "COG",
    "L_MIDDLEA": "L_WRIST",
    "L_MIDDLEB": "L_MIDDLEA",
    "L_MIDDLEC": "L_MIDDLEB",
    "L_MIDDLE_END": "L_MIDDLEC",
    "L_MIDSUPPORT": "L_MIDAXLE",
    "L_MIDWHEEL": "MIDAXLE",
    "L_OUTERFLAP": "L_THRUSTER",
    "L_OUTERTHRUSTERPANELA": "L_THRUSTER",
    "L_OUTERTHRUSTERPANELB": "L_THRUSTER",
    "L_OUTERTHRUSTERPANELC": "L_THRUSTER",
    "L_OVERHEADDOOR1": "COG",
    "L_OVERHEADDOOR2": "L_OVERHEADDOOR1",
    "L_PINKYA": "L_WRIST",
    "L_PINKYB": "L_PINKYA",
    "L_PINKYC": "L_PINKYB",
    "L_PINKY_END": "L_PINKYC",
    "L_PITCHRECOIL": "",
    "L_REARAXLE": "COG",
    "L_REARDOOR": "COG",
    "L_REARFLAP": "L_HIP",
    "L_REARGEAR1": "COG",
    "L_REARGEAR2": "L_REARGEAR1",
    "L_REARGEAR3": "L_REARGEAR2",
    "L_REARGEAR4": "L_REARGEAR3",
    "L_REARGEARA": "COG",
    "L_REARGEARB": "L_REARGEARA",
    "L_REARGEARC": "L_REARGEARB",
    "L_REARGEARDOOR": "COG",
    "L_REARGEARDOOR1": "COG",
    "L_REARGEARDOOR2": "COG",
    "L_REARGEARSUPPORT": "COG",
    "L_REARSHOCKLOWER": "L_REARSUSPENSION",
    "L_REARSHOCKUPPER": "COG",
    "L_REARSUPPORT": "L_REARAXLE",
    "L_REARSUSPENSION": "COG",
    "L_REARTHRUSTER": "COG",
    "L_REARTHRUSTER1": "COG",
    "L_REARTHRUSTER2": "L_REARTHRUSTER1",
    "L_REARTHRUSTERFLAP": "L_REARTHRUSTER",
    "L_REARVENTS": "COG",
    "L_REARWHEEL": "REARAXLE",
    "L_REARWHEELPIVOT": "L_REARSUSPENSION",
    "L_RINGA": "L_WRIST",
    "L_RINGB": "L_RINGA",
    "L_RINGC": "L_RINGB",
    "L_RING_END": "L_RINGC",
    "L_RUDDER": "TAIL",
    "L_SHOCKLOWER": "",
    "L_SHOCKUPPER": "",
    "L_SHOULDER": "L_CLAVICLE",
    "L_SHOULDERROLL": "L_SHOULDER",
    "L_SIDELOCK_BACK": "",
    "L_SIDELOCK_BACK_END": "L_SIDELOCK_BACK",
    "L_SIDELOCK_FRONT": "",
    "L_SIDELOCK_FRONT_END": "L_SIDELOCK_FRONT",
    "L_SIDEVENT1": "COG",
    "L_SIDEVENT2": "COG",
    "L_SIDEVENT3": "COG",
    "L_SIDEVENT4": "COG",
    "L_SMOKESTACK": "",
    "L_SPIKEINSIDE": "",
    "L_SPIKEOUTSIDE": "",
    "L_STEERINGLINK": "",
    "L_STEERINGLINKINNER": "COG",
    "L_STEERINGLINKOUTER": "L_STEERINGPIVOT",
    "L_STEERINGPIVOT": "L_FRONTSUSPENSION",
    "L_SUPPORT1": "COG",
    "L_SUPPORT2": "L_SUPPORT1",
    "L_SUPPORT3": "L_SUPPORT2",
    "L_SUPPORT4": "L_SUPPORT3",
    "L_THRUSTER": "COG",
    "L_THRUSTER1": "COG",
    "L_THRUSTER2": "L_THRUSTER1",
    "L_THRUSTER3": "L_THRUSTER2",
    "L_THRUSTERFLAP": "L_THRUSTER1",
    "L_THRUSTERFLAP1": "L_THRUSTER",
    "L_THRUSTERFLAP2": "L_THRUSTER",
    "L_THRUSTERPANEL": "COG",
    "L_THUMBA": "L_WRIST",
    "L_THUMBB": "L_THUMBA",
    "L_THUMBC": "L_THUMBB",
    "L_THUMB_END": "L_THUMBC",
    "L_TOE": "L_BALL",
    "L_TOPCANISTER": "",
    "L_TOPHOUSING": "",
    "L_TOPTUBE": "",
    "L_TOP_BOX01": "",
    "L_TOP_BOX02": "",
    "L_TOP_BOX03": "",
    "L_TOP_BOX04": "",
    "L_TOP_CANNISTER01": "",
    "L_TOP_CANNISTER02": "",
    "L_TOP_CANNISTER03": "",
    "L_UPPERSUSPENSION": "",
    "L_VENT": "",
    "L_VENT_END": "L_VENT",
    "L_WEAPON": "L_WRIST",
    "L_WHEEL1": "COG",
    "L_WHEEL10_ROTATE": "COG",
    "L_WHEEL10_TRANSLATE": "COG",
    "L_WHEEL11_ROTATE": "COG",
    "L_WHEEL11_TRANSLATE": "COG",
    "L_WHEEL12_ROTATE": "COG",
    "L_WHEEL12_TRANSLATE": "COG",
    "L_WHEEL13_TRANSLATE": "COG",
    "L_WHEEL1_DEPLOYHUB": "L_WHEEL1_ROTATE",
    "L_WHEEL1_DEPLOYSPIKE1": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE2": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE3": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE4": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE5": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE6": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE7": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_DEPLOYSPIKE8": "L_WHEEL1_DEPLOYHUB",
    "L_WHEEL1_ROTATE": "COG",
    "L_WHEEL1_TRANSLATE": "COG",
    "L_WHEEL2": "COG",
    "L_WHEEL2_ROTATE": "COG",
    "L_WHEEL2_TRANSLATE": "COG",
    "L_WHEEL3": "COG",
    "L_WHEEL3_ROTATE": "COG",
    "L_WHEEL3_TRANSLATE": "COG",
    "L_WHEEL4": "COG",
    "L_WHEEL4_ROTATE": "COG",
    "L_WHEEL4_TRANSLATE": "COG",
    "L_WHEEL5": "COG",
    "L_WHEEL5_ROTATE": "COG",
    "L_WHEEL5_TRANSLATE": "COG",
    "L_WHEEL6": "COG",
    "L_WHEEL6_DEPLOYHUB": "L_WHEEL6_ROTATE",
    "L_WHEEL6_DEPLOYSPIKE1": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE2": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE3": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE4": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE5": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE6": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE7": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_DEPLOYSPIKE8": "L_WHEEL6_DEPLOYHUB",
    "L_WHEEL6_ROTATE": "COG",
    "L_WHEEL6_TRANSLATE": "COG",
    "L_WHEEL7": "COG",
    "L_WHEEL7_ROTATE": "COG",
    "L_WHEEL7_TRANSLATE": "COG",
    "L_WHEEL8": "COG",
    "L_WHEEL8_ROTATE": "COG",
    "L_WHEEL8_TRANSLATE": "COG",
    "L_WHEEL9_ROTATE": "COG",
    "L_WHEEL9_TRANSLATE": "COG",
    "L_WING": "COG",
    "L_WINGFAN": "L_WING",
    "L_WINGVENTS": "L_THRUSTER",
    "L_WRIST": "L_FOREARM",
    "MIDAXLE": "COG",
    "MUZZLE1_SOCKET": "",
    "NECK": "SPINEUPPER",
    "PELVIS": "COG",
    "PITCH": "YAW",
    "PITCHRECOIL": "PITCH",
    "REARAXLE": "COG",
    "REARFANPIVOT": "",
    "REARFANTHRUST": "",
    "REARFANYAW": "",
    "REARFLAPS": "COG",
    "REARGEAR1": "COG",
    "REARGEAR2": "REARGEAR1",
    "REARGEAR3": "REARGEAR2",
    "REARGEAR4": "REARGEAR3",
    "REARGEAR5": "COG",
    "REARHATCHA": "COG",
    "REARHATCHB": "REARHATCHA",
    "REARSHOCKLOWER": "",
    "REARSHOCKUPPER": "",
    "REARSUPPORT1": "COG",
    "REARSUPPORT2": "REARSUPPORT1",
    "REARSUPPORT3": "REARSUPPORT2",
    "REARSUPPORT4": "REARSUPPORT3",
    "REARSUPPORT5": "REARSUPPORT4",
    "REARSUSPENSION": "",
    "REARTHRUSTER": "COG",
    "R_ANKLE": "R_KNEE",
    "R_BACKPLATE": "",
    "R_BACKSHOCKA": "",
    "R_BACKSHOCKA_TOP": "R_BACKSHOCKA",
    "R_BACKSHOCKB": "",
    "R_BACKSHOCKB_TOP": "R_BACKSHOCKB",
    "R_BACK_CANNISTER01": "",
    "R_BACK_CANNISTER02": "",
    "R_BALL": "R_ANKLE",
    "R_BARREL": "",
    "R_BARRELFRONT_EXTEND": "",
    "R_BARRELSIDELAYER1": "",
    "R_BARRELSIDELAYER1_END": "R_BARRELSIDELAYER1",
    "R_BARRELSIDELAYER2": "",
    "R_BARRELSIDE_EXTEND": "",
    "R_BOTTOMCANISTER": "",
    "R_BOTTOMHOUSING": "",
    "R_BOTTOMTUBE": "",
    "R_BOTTOMVENTS": "COG",
    "R_CLAVICLE": "SPINEUPPER",
    "R_CYLINDERDN": "",
    "R_CYLINDEREXTEND_END": "R_CYLINDEREXTEND",
    "R_CYLINDEREXTEND_END1": "",
    "R_CYLINDERUP": "",
    "R_DEPLOY_MECANISM": "",
    "R_DEPLOY_MECANISM_END": "R_DEPLOY_MECANISM",
    "R_ELBOW": "R_SHOULDERROLL",
    "R_FAN": "R_THRUSTER",
    "R_FLAP": "",
    "R_FLAPPER": "COG",
    "R_FOOT": "",
    "R_FOOT_INSIDETOES": "",
    "R_FOOT_INSIDETOES_END": "R_FOOT_INSIDETOES",
    "R_FOOT_OUTSIDETOES": "",
    "R_FOOT_OUTSIDETOES_END": "R_FOOT_OUTSIDETOES",
    "R_FOOT_SPIKES": "",
    "R_FOOT_SPIKES_END": "R_FOOT_SPIKES",
    "R_FOREARM": "R_ELBOW",
    "R_FRONTAXLE": "STEERING",
    "R_FRONTDOOR": "COG",
    "R_FRONTFLAP": "R_HIP",
    "R_FRONTGEAR1": "COG",
    "R_FRONTGEAR2": "R_FRONTGEAR1",
    "R_FRONTGEAR3": "R_FRONTGEAR2",
    "R_FRONTGEAR4": "R_FRONTGEAR3",
    "R_FRONTGEAR5": "COG",
    "R_FRONTGEARDOOR1": "COG",
    "R_FRONTGEARDOOR2": "COG",
    "R_FRONTPLATE": "",
    "R_FRONTSHOCKA": "",
    "R_FRONTSHOCKA_TOP": "",
    "R_FRONTSHOCKB": "",
    "R_FRONTSHOCKB_TOP": "",
    "R_FRONTSHOCKLOWER": "R_STEERINGPIVOT",
    "R_FRONTSHOCKUPPER": "COG",
    "R_FRONTSUPPORT": "R_FRONTAXLE",
    "R_FRONTSUPPORT1": "",
    "R_FRONTSUPPORT2": "",
    "R_FRONTSUPPORT3": "",
    "R_FRONTSUPPORT4": "",
    "R_FRONTSUSPENSION": "COG",
    "R_FRONTWHEEL": "FRONTAXLE",
    "R_FRONTWHEEL_DUMMY": "R_FRONTSUPPORT",
    "R_GEARA": "COG",
    "R_GEARB": "R_GEARA",
    "R_GEARC": "R_GEARB",
    "R_GEARDOOR": "COG",
    "R_GEARSUPPORTA": "R_GEARA",
    "R_GEARSUPPORTB": "COG",
    "R_HATCH_BOTTOM": "",
    "R_HATCH_BOTTOM_END": "R_HATCH_BOTTOM",
    "R_HATCH_TOP": "",
    "R_HATCH_TOP_END": "R_HATCH_TOP",
    "R_HIP": "PELVIS",
    "R_INDEXA": "R_WRIST",
    "R_INDEXB": "R_INDEXA",
    "R_INDEXC": "R_INDEXB",
    "R_INDEX_END": "R_INDEXC",
    "R_INNERFLAP": "R_THRUSTER",
    "R_INNERTHRUSTER": "R_THRUSTER",
    "R_KNEE": "R_HIP",
    "R_KNOB_ROTATE": "",
    "R_KNOB_TRANSLATE": "",
    "R_LEG": "",
    "R_LEG_PISTON": "",
    "R_LOWERSUSPENSION": "",
    "R_MIDAXLE": "COG",
    "R_MIDDLEA": "R_WRIST",
    "R_MIDDLEB": "R_MIDDLEA",
    "R_MIDDLEC": "R_MIDDLEB",
    "R_MIDDLE_END": "R_MIDDLEC",
    "R_MIDSUPPORT": "R_MIDAXLE",
    "R_MIDWHEEL": "MIDAXLE",
    "R_OUTERFLAP": "R_THRUSTER",
    "R_OUTERTHRUSTERPANELA": "R_THRUSTER",
    "R_OUTERTHRUSTERPANELB": "R_THRUSTER",
    "R_OUTERTHRUSTERPANELC": "R_THRUSTER",
    "R_OVERHEADDOOR1": "COG",
    "R_OVERHEADDOOR2": "R_OVERHEADDOOR1",
    "R_PINKYA": "R_WRIST",
    "R_PINKYB": "R_PINKYA",
    "R_PINKYC": "R_PINKYB",
    "R_PINKY_END": "R_PINKYC",
    "R_PITCHRECOIL": "",
    "R_REARAXLE": "COG",
    "R_REARDOOR": "COG",
    "R_REARFLAP": "R_HIP",
    "R_REARGEAR1": "COG",
    "R_REARGEAR2": "R_REARGEAR1",
    "R_REARGEAR3": "R_REARGEAR2",
    "R_REARGEAR4": "R_REARGEAR3",
    "R_REARGEARA": "COG",
    "R_REARGEARB": "R_REARGEARA",
    "R_REARGEARC": "R_REARGEARB",
    "R_REARGEARDOOR": "COG",
    "R_REARGEARDOOR1": "COG",
    "R_REARGEARDOOR2": "COG",
    "R_REARGEARSUPPORT": "COG",
    "R_REARSHOCKLOWER": "R_REARSUSPENSION",
    "R_REARSHOCKUPPER": "COG",
    "R_REARSUPPORT": "R_REARAXLE",
    "R_REARSUSPENSION": "COG",
    "R_REARTHRUSTER": "COG",
    "R_REARTHRUSTER1": "COG",
    "R_REARTHRUSTER2": "R_REARTHRUSTER1",
    "R_REARTHRUSTERFLAP": "R_REARTHRUSTER",
    "R_REARVENTS": "COG",
    "R_REARWHEEL": "REARAXLE",
    "R_REARWHEELPIVOT": "R_REARSUSPENSION",
    "R_RINGA": "R_WRIST",
    "R_RINGB": "R_RINGA",
    "R_RINGC": "R_RINGB",
    "R_RING_END": "R_RINGC",
    "R_RUDDER": "TAIL",
    "R_SHOCKLOWER": "",
    "R_SHOCKUPPER": "",
    "R_SHOULDER": "R_CLAVICLE",
    "R_SHOULDERROLL": "R_SHOULDER",
    "R_SIDELOCK_BACK": "",
    "R_SIDELOCK_BACK_END": "R_SIDELOCK_BACK",
    "R_SIDELOCK_FRONT": "",
    "R_SIDELOCK_FRONT_END": "R_SIDELOCK_FRONT",
    "R_SIDEVENT1": "COG",
    "R_SIDEVENT2": "COG",
    "R_SIDEVENT3": "COG",
    "R_SIDEVENT4": "COG",
    "R_SMOKESTACK": "",
    "R_SPIKEINSIDE": "",
    "R_SPIKEOUTSIDE": "",
    "R_STEERINGLINK": "",
    "R_STEERINGLINKINNER": "COG",
    "R_STEERINGLINKOUTER": "R_STEERINGPIVOT",
    "R_STEERINGPIVOT": "R_FRONTSUSPENSION",
    "R_SUPPORT1": "COG",
    "R_SUPPORT2": "R_SUPPORT1",
    "R_SUPPORT3": "R_SUPPORT2",
    "R_SUPPORT4": "R_SUPPORT3",
    "R_THRUSTER": "COG",
    "R_THRUSTER1": "COG",
    "R_THRUSTER2": "R_THRUSTER1",
    "R_THRUSTER3": "R_THRUSTER2",
    "R_THRUSTERFLAP": "R_THRUSTER1",
    "R_THRUSTERFLAP1": "R_THRUSTER",
    "R_THRUSTERFLAP2": "R_THRUSTER",
    "R_THRUSTERPANEL": "COG",
    "R_THUMBA": "R_WRIST",
    "R_THUMBB": "R_THUMBA",
    "R_THUMBC": "R_THUMBB",
    "R_THUMB_END": "R_THUMBC",
    "R_TOE": "R_BALL",
    "R_TOPCANISTER": "",
    "R_TOPHOUSING": "",
    "R_TOPTUBE": "",
    "R_TOP_BOX01": "",
    "R_TOP_BOX02": "",
    "R_TOP_BOX03": "",
    "R_TOP_BOX04": "",
    "R_TOP_CANNISTER01": "",
    "R_TOP_CANNISTER02": "",
    "R_TOP_CANNISTER03": "",
    "R_UPPERSUSPENSION": "",
    "R_VENT": "",
    "R_VENT_END": "R_VENT",
    "R_WEAPON": "R_WRIST",
    "R_WHEEL1": "COG",
    "R_WHEEL10_ROTATE": "COG",
    "R_WHEEL10_TRANSLATE": "COG",
    "R_WHEEL11_ROTATE": "COG",
    "R_WHEEL11_TRANSLATE": "COG",
    "R_WHEEL12_ROTATE": "COG",
    "R_WHEEL12_TRANSLATE": "COG",
    "R_WHEEL13_TRANSLATE": "COG",
    "R_WHEEL1_DEPLOYHUB": "R_WHEEL1_ROTATE",
    "R_WHEEL1_DEPLOYSPIKE1": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE2": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE3": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE4": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE5": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE6": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE7": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_DEPLOYSPIKE8": "R_WHEEL1_DEPLOYHUB",
    "R_WHEEL1_ROTATE": "COG",
    "R_WHEEL1_TRANSLATE": "COG",
    "R_WHEEL2": "COG",
    "R_WHEEL2_ROTATE": "COG",
    "R_WHEEL2_TRANSLATE": "COG",
    "R_WHEEL3": "COG",
    "R_WHEEL3_ROTATE": "COG",
    "R_WHEEL3_TRANSLATE": "COG",
    "R_WHEEL4": "COG",
    "R_WHEEL4_ROTATE": "COG",
    "R_WHEEL4_TRANSLATE": "COG",
    "R_WHEEL5": "COG",
    "R_WHEEL5_ROTATE": "COG",
    "R_WHEEL5_TRANSLATE": "COG",
    "R_WHEEL6": "COG",
    "R_WHEEL6_DEPLOYHUB": "R_WHEEL6_ROTATE",
    "R_WHEEL6_DEPLOYSPIKE1": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE2": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE3": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE4": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE5": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE6": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE7": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_DEPLOYSPIKE8": "R_WHEEL6_DEPLOYHUB",
    "R_WHEEL6_ROTATE": "COG",
    "R_WHEEL6_TRANSLATE": "COG",
    "R_WHEEL7": "COG",
    "R_WHEEL7_ROTATE": "COG",
    "R_WHEEL7_TRANSLATE": "COG",
    "R_WHEEL8": "COG",
    "R_WHEEL8_ROTATE": "COG",
    "R_WHEEL8_TRANSLATE": "COG",
    "R_WHEEL9_ROTATE": "COG",
    "R_WHEEL9_TRANSLATE": "COG",
    "R_WING": "COG",
    "R_WINGFAN": "",
    "R_WINGVENTS": "R_THRUSTER",
    "R_WRIST": "R_FOREARM",
    "SHELL1": "",
    "SHELL10": "",
    "SHELL11": "",
    "SHELL12": "",
    "SHELL13": "",
    "SHELL14": "",
    "SHELL2": "",
    "SHELL3": "",
    "SHELL4": "",
    "SHELL5": "",
    "SHELL6": "",
    "SHELL7": "",
    "SHELL9": "",
    "SPINELOWER": "PELVIS",
    "SPINEMIDDLE": "SPINELOWER",
    "SPINEUPPER": "SPINEMIDDLE",
    "STEERING": "COG",
    "TAIL": "COG",
    "TAILVENTS": "TAIL",
    "TRAJECTORY": "",
    "WORLDROOT": None,
    "YAW": "WORLDROOT",
    "YAWRECOIL": "YAW",
}

with open("bonelist.txt") as f:           
    bones = f.read().upper().split()

uniq_bones = sorted(list(set(bones)))
with open("ps2_bone_map.py", "w") as f:
    f.write("BONE_HASHMAP = {\n")
    for bone in uniq_bones:
        f.write(f'    {oaat(bone.encode("utf-8"))}: "{bone}",\n')
    f.write("}\n\n")
    f.write("HIERARCHY = {\n")
    for bone in uniq_bones:
        if bone in HIERARCHY and HIERARCHY[bone] is not None:
            f.write(f'    "{bone}": "{HIERARCHY[bone]}",\n')
        elif bone.endswith("_END"):
            f.write(f'    "{bone}": "{bone[:-4]}",\n')
        elif bone in HIERARCHY and HIERARCHY[bone] is None:
            f.write(f'    "{bone}": None,\n')
        else:
            f.write(f'    "{bone}": "",\n')

    f.write("}\n\n")