"""
================================================================================
BioDINO-Edge: Domain-Specific Taxonomic Knowledge & Multi-Prompt Ensembles
================================================================================
Curated expert botanical descriptions coupling genus, species, and pathogenic
etiology for zero-shot text prototype synthesis via BioCLIP ViT-B.
================================================================================
"""

PLANTVILLAGE_TAXONOMIC_PROMPTS = {
    "Apple___Apple_scab": [
        "A botanical photograph of Apple (Malus domestica) foliage with olive-green velvety lesions caused by Venturia inaequalis.",
        "Apple scab leaf symptom showing dark circular olive-black scabby fungal spots on leaf surface.",
        "Malus leaf infected with Venturia scab fungal lesions with irregular margins."
    ],
    "Apple___Black_rot": [
        "Apple foliage showing Black rot frogeye leaf spots with purple margins and concentric zones caused by Botryosphaeria obtusa.",
        "Circular brown necrotic lesions with dark border on Malus domestica leaf.",
        "Frogeye leaf spot on apple leaf blade with distinct tan center."
    ],
    "Apple___Cedar_apple_rust": [
        "Apple leaf exhibiting bright yellow-orange circular rust spots with fungal aecia caused by Gymnosporangium juniperi-virginianae.",
        "Bright orange-yellow cedar apple rust lesions on upper leaf surface of Malus domestica.",
        "Characteristic yellow chlorotic rust spots with small dark pycnia dots on apple leaf."
    ],
    "Apple___healthy": [
        "A botanical photograph of healthy vigorous foliage of Apple (Malus domestica) showing clean vibrant green leaves.",
        "Flawless healthy green apple leaf blade displaying uniform emerald lamina and intact smooth venation.",
        "Clean vigorous vegetative green foliage of Malus domestica fruit tree with pristine leaf surface."
    ],
    "Grape___Black_rot": [
        "Grape (Vitis vinifera) foliage showing Black rot reddish-brown circular necrotic spots with minute black pycnidia caused by Guignardia bidwellii.",
        "Circular necrotic spots with dark margins and tiny black specks of fungal fruiting bodies on grape leaf.",
        "Guignardia black rot leaf spot on grape showing distinct circular boundaries and dead brown center."
    ],
    "Grape___Esca_(Black_Measles)": [
        "Grape leaf showing Esca complex Black Measles with interveinal chlorosis and necrosis creating a distinct tiger-stripe pattern.",
        "Tiger-stripe foliar symptoms on Vitis vinifera with yellow and necrotic bands between green veins caused by Phaeomoniella chlamydospora.",
        "Severe interveinal scorching and tiger-striping on grape foliage with green tissue bordering veins."
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Grape leaf exhibiting Isariopsis leaf blight showing irregular angular dark brown foliar lesions caused by Pseudocercospora vitis.",
        "Angular brown necrotic leaf spots bounded by leaf veinlets on Vitis vinifera foliage.",
        "Dark brown irregular angular foliar blight lesions on grape leaf."
    ],
    "Grape___healthy": [
        "A botanical photograph of healthy vigorous foliage of Grape (Vitis vinifera) showing clean vibrant green leaves.",
        "Flawless healthy green grape leaf blade displaying intact lobed structure and rich chlorophyll green coloration.",
        "Clean vigorous vegetative green foliage of Vitis vinifera vine with pristine leaf surface."
    ],
    "Potato___Early_blight": [
        "Potato foliage showing dark brown circular necrotic spots with concentric target-board rings caused by Alternaria solani.",
        "Target-board concentric ring lesions of early blight on Solanum tuberosum foliage.",
        "Dark brown target-spot blight on potato leaf blade surrounded by yellow chlorotic halo."
    ],
    "Potato___Late_blight": [
        "Potato foliage with dark water-soaked greasy necrotic lesions and pale green margins caused by Phytophthora infestans.",
        "Destructive late blight necrotic foliar decay on Solanum tuberosum leaf blade.",
        "Water-soaked dark brown late blight lesions with white fungal mildew margins on potato leaf."
    ],
    "Potato___healthy": [
        "A botanical photograph of healthy vigorous foliage of Potato (Solanum tuberosum) showing clean vibrant green leaflets.",
        "Flawless healthy green compound potato leaf blade displaying uniform emerald lamina and intact venation.",
        "Clean vigorous vegetative green foliage of Solanum tuberosum plant with pristine leaflet surface."
    ]
}

PLANTDOC_TAXONOMIC_PROMPTS = {
    "Apple_Scab_Leaf": [
        "Apple leaf with dark olive-green velvety fungal scab spots caused by Venturia inaequalis.",
        "A photo of apple foliage exhibiting irregular scabby dark lesions on the leaf blade.",
        "Malus domestica leaf blade affected by apple scab fungal infection."
    ],
    "Apple_leaf": [
        "A healthy vibrant green leaf of apple tree (Malus domestica) showing clean lamina.",
        "Clean, flawless green foliage of healthy vigorous apple plant.",
        "Flawless intact green apple leaf displaying healthy mesophyll."
    ],
    "Apple_rust_leaf": [
        "Apple foliage with bright orange circular cedar apple rust lesions caused by Gymnosporangium.",
        "Bright yellow-orange rust pustules on Malus domestica leaf surface.",
        "Characteristic cedar apple rust chlorotic spots on apple leaf blade."
    ],
    "Bell_pepper_leaf": [
        "Healthy vigorous green leaf of sweet bell pepper plant (Capsicum annuum).",
        "Clean vibrant emerald foliage of bell pepper displaying intact smooth leaf blade.",
        "Flawless vigorous foliage of healthy Capsicum annuum."
    ],
    "Bell_pepper_leaf_spot": [
        "Bell pepper leaf showing dark water-soaked bacterial leaf spot lesions caused by Xanthomonas.",
        "Small necrotic spots with yellow chlorotic halos on sweet pepper foliage.",
        "Bacterial spot symptoms on Capsicum annuum leaf."
    ],
    "Blueberry_leaf": [
        "Healthy green leaf of blueberry bush (Vaccinium corymbosum).",
        "Clean vigorous foliage of blueberry plant showing intact oval leaf blade.",
        "Healthy pristine blueberry leaf blade with vibrant green cuticle."
    ],
    "Cherry_leaf": [
        "Healthy vibrant green leaf of sweet cherry tree (Prunus avium).",
        "Flawless cherry foliage showing smooth glossy green surface.",
        "Intact healthy Prunus leaf blade with uniform coloration."
    ],
    "Corn_Gray_leaf_spot": [
        "Corn leaf showing rectangular tan to gray necrotic lesions bounded by veins caused by Cercospora zeae-maydis.",
        "Gray leaf spot lesions with straight parallel borders on maize foliage.",
        "Rectangular gray necrotic blights on Zea mays leaf blade."
    ],
    "Corn_leaf_blight": [
        "Corn foliage with elongated cigar-shaped grayish-green necrotic lesions caused by Exserohilum turcicum.",
        "Northern corn leaf blight showing large elliptical lesions on maize leaf.",
        "Cigar-shaped foliar blights on corn leaf blade."
    ],
    "Corn_rust_leaf": [
        "Corn leaf blade covered with reddish-brown powdery rust pustules caused by Puccinia sorghi.",
        "Common rust symptoms showing golden-brown fungal spore pustules on maize leaf.",
        "Puccinia rust lesions rupturing the corn leaf epidermis."
    ],
    "Peach_leaf": [
        "Healthy elongated lanceolate green leaf of peach tree (Prunus persica).",
        "Clean vibrant foliage of healthy peach fruit tree.",
        "Vibrant green peach leaf displaying intact smooth surface."
    ],
    "Potato_leaf_early_blight": [
        "Potato leaf showing circular brown necrotic spots with concentric rings caused by Alternaria solani.",
        "Target-board concentric ring lesions of early blight on Solanum tuberosum foliage.",
        "Dark brown target-spot blight on potato leaf blade."
    ],
    "Potato_leaf_late_blight": [
        "Potato foliage with dark water-soaked greasy lesions and white mold margins caused by Phytophthora infestans.",
        "Destructive late blight necrotic foliar decay on potato leaf blade.",
        "Water-soaked dark brown late blight lesions on Solanum tuberosum leaf."
    ],
    "Raspberry_leaf": [
        "Healthy serrated green compound leaf of raspberry plant (Rubus idaeus).",
        "Clean vigorous foliage of raspberry displaying intact green leaflets.",
        "Flawless green raspberry leaf blade with uniform coloration."
    ],
    "Soyabean_leaf": [
        "Healthy green trifoliate leaf blade of soybean plant (Glycine max).",
        "Clean vigorous soybean foliage displaying intact emerald leaflets.",
        "Flawless healthy Glycine max leaf with uniform mesophyll."
    ],
    "Squash_Powdery_mildew_leaf": [
        "Squash leaf blade covered in white powdery talc-like fungal patches caused by Podosphaera xanthii.",
        "Powdery mildew white fungal coating covering the upper surface of cucurbit leaf.",
        "Severe white powdery foliar mildew on squash leaf."
    ],
    "Strawberry_leaf": [
        "Healthy trifoliate compound green leaf of strawberry plant (Fragaria ananassa).",
        "Clean vibrant foliage of strawberry displaying intact serrated leaflets.",
        "Flawless healthy Fragaria leaf blade with rich green color."
    ],
    "Tomato_Early_blight_leaf": [
        "Tomato leaf showing brown necrotic spots with concentric target-board rings caused by Alternaria linariae.",
        "Concentric ring target lesions surrounded by chlorotic yellow halo on tomato foliage.",
        "Alternaria early blight foliar symptoms on Solanum lycopersicum."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Tomato foliage densely covered in small circular gray lesions with dark margins caused by Septoria lycopersici.",
        "Numerous small circular spots with grayish-white centers and black speck pycnidia on tomato leaf.",
        "Septoria leaf spot lesions defoliating tomato leaf blade."
    ],
    "Tomato_leaf": [
        "Healthy vigorous compound green leaf of tomato plant (Solanum lycopersicum).",
        "Clean flawless green foliage of tomato displaying intact leaflets.",
        "Flawless vibrant tomato leaf blade with uniform emerald coloration."
    ],
    "Tomato_leaf_bacterial_spot": [
        "Tomato leaf blade exhibiting small dark brown water-soaked angular spots caused by Xanthomonas.",
        "Bacterial leaf spot lesions with yellowish halos on tomato foliage.",
        "Xanthomonas bacterial speck and spot on Solanum lycopersicum leaf."
    ],
    "Tomato_leaf_late_blight": [
        "Tomato foliage showing extensive dark water-soaked necrotic lesions caused by Phytophthora infestans.",
        "Rapidly expanding greasy dark lesions of late blight on tomato leaves.",
        "Severe destructive foliar blight caused by Phytophthora on tomato."
    ],
    "Tomato_leaf_mosaic_virus": [
        "Tomato leaf blade exhibiting mottled light and dark green mosaic patterns and leaf curling caused by Tomato mosaic virus.",
        "Mosaic virus foliar distortion with blistered light green and dark green foliage on tomato.",
        "ToMV viral symptoms showing mottled mosaic discoloration on tomato leaf."
    ],
    "Tomato_leaf_yellow_virus": [
        "Tomato foliage showing severe upward cupping, leaf curling and marginal chlorosis caused by Tomato yellow leaf curl virus.",
        "Stunted curled yellow foliage of TYLCV viral infection on tomato plant.",
        "Severe chlorotic yellowing and upward rolling of tomato leaf margins."
    ],
    "Tomato_mold_leaf": [
        "Tomato leaf showing yellow chlorotic patches on top and olive-green velvety mold on the underside caused by Passalora fulva.",
        "Leaf mold fungal symptoms showing olive-brown velvety growth under tomato foliage.",
        "Passalora leaf mold on Solanum lycopersicum leaf blade."
    ],
    "grape_leaf": [
        "Healthy smooth lobed green foliage of grape vine (Vitis vinifera).",
        "Clean vigorous grape leaf displaying intact lobed leaf blade.",
        "Flawless healthy Vitis vinifera leaf with rich green lamina."
    ],
    "grape_leaf_black_rot": [
        "Grape leaf blade showing reddish-brown circular necrotic spots with black pycnidia caused by Guignardia bidwellii.",
        "Black rot lesions with distinct circular margins and tiny fungal fruiting specks on grape foliage.",
        "Guignardia black rot leaf spot on Vitis vinifera leaf."
    ]
}
