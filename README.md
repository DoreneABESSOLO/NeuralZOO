# NeuralZOO

## 1] Introduction.

## 2] Veille technologique

### 2.1] Multilayer Perceptron

#### 2.1.1] Perceptron Multicouches, architectures & hyperparamètres  
Le Perceptron Multicouches (MLP) représente une évolution significative dans le domaine des réseaux de neurones artificiels, marquant un progrès notable depuis les premiers modèles de perceptrons simples. Développé dans les années 1980, le MLP a été conçu pour surmonter les limitations des perceptrons monocouches en introduisant des architectures plus complexes et des capacités d'apprentissage plus profondes.  
Initialement inspiré par les neurones biologiques et les premières théories du réseau neuronal formel, le concept de MLP a été formalisé pour permettre l'apprentissage de modèles non linéaires grâce à l'ajout de multiples couches de neurones. Cette approche a permis aux réseaux de neurones de traiter efficacement des problèmes de classification et de régression complexes, en exploitant des fonctions d'activation non linéaires dans les couches cachées.  

Le MLP est un type de réseau de neurones artificiels composé de plusieurs couches:  
- Couches d'entrée  
- Couches cachées  
- Couches de sortie  
- Couches denses  

<u>Couches d'entrée:</u>  
Les couches d'entrée dans un MLP jouent un rôle crucial dans la réception et la transmission des données initiales vers le réseau de neurones. Chaque neurone dans cette couche correspond à une caractéristique spécifique des données en entrée. Par exemple, dans le cas d'une tâche de classification d'images, chaque neurone pourrait représenter les valeurs de pixels pour une zone particulière de l'image. Les valeurs initiales sont propagées à travers le réseau pour subir des transformations et des calculs ultérieurs.  

<u>Couches cachées:</u>  
Les couches cachées constituent le cœur du MLP, où la majorité du traitement et de l'apprentissage se produisent. Chaque couche cachée est composée de plusieurs neurones, chacun calculant une combinaison linéaire des sorties de la couche précédente, suivie d'une fonction d'activation non linéaire. Cela permet au réseau de capturer des motifs et des relations complexes dans les données, ce qui est crucial pour la capacité du modèle à généraliser à de nouvelles observations. L'ajout de plusieurs couches cachées permet au MLP d'apprendre des représentations hiérarchiques et abstraites des données.  

<u>Couches de sortie:</u>  
Les couches de sortie dans un MLP produisent les prédictions finales du modèle après avoir traité les informations à travers les couches d'entrée et les couches cachées. Le nombre de neurones dans cette couche dépend de la nature spécifique du problème :  
- Pour une tâche de classification binaire, il y aurait un seul neurone avec une fonction d'activation comme sigmoid, indiquant la probabilité d'appartenir à une classe.  
- Pour la classification multiclasse, il y aurait plusieurs neurones, un pour chaque classe, avec une activation comme softmax pour obtenir des probabilités de classe.  
- Pour la régression, il y aurait généralement un seul neurone produisant une valeur continue.  

<u>Couches denses:</u> 
Les couches denses dans un MLP, souvent appelées couches entièrement connectées, impliquent que chaque neurone est connecté à chaque neurone de la couche précédente et suivante. Cela crée une structure de réseau dense où toutes les entrées sont connectées à toutes les sorties, facilitant ainsi la propagation et le calcul efficaces des valeurs à travers le réseau. Cette connectivité complète permet au modèle d'apprendre des relations complexes entre les caractéristiques d'entrée et les cibles de sortie, tout en nécessitant un ajustement minutieux des poids et des biais pour optimiser les performances du modèle.  

En intégrant ces différentes couches dans une architecture multicouche, le Perceptron Multicouches peut capturer des modèles non linéaires complexes dans les données, améliorant ainsi sa capacité à résoudre une variété de problèmes de machine learning. 

<u>**Les hyperparamètres**</u>
Les hyperparamètres jouent un rôle essentiel dans la conception et l'entraînement des réseaux de neurones, y compris les Perceptrons Multicouches (MLP). Ce sont des paramètres dont les valeurs sont fixées avant le début du processus d'apprentissage et qui influencent directement la performance et la capacité du modèle à apprendre et généraliser à de nouvelles données.

Choisir les bons hyperparamètres est crucial car ils déterminent la structure du réseau neuronal, la manière dont il apprend à partir des données, et la qualité des prédictions qu'il produit. Par exemple, le nombre de couches cachées et le nombre de neurones par couche définissent la complexité du modèle, tandis que le choix de la fonction d'activation et de la fonction de perte impacte la capacité du réseau à modéliser des relations non linéaires et à minimiser l'erreur de prédiction.  

Pour un MLP, les hyperparamètres clés sont les suivants:  

**Fonction d'activation:** Elles introduisent de la non-linéarité dans le modèle, permettant au réseau de modéliser des relations non linéaires dans les données. Les choix courants incluent ReLU (Rectified Linear Unit), sigmoid, tanh, etc. ReLU est souvent préférée pour sa simplicité et sa performance dans de nombreux cas, mais le choix dépend du problème spécifique et de la nature des données.  

**Fonction de perte:** Elle mesure la différence entre les valeurs prédites par le modèle et les valeurs réelles lors de l'entraînement. Elle guide l'optimisation des poids du réseau pendant la rétropropagation. Pour différents types de tâches (classification, régression), des fonctions de perte spécifiques comme la cross-entropy pour la classification ou le Mean Squared Error (MSE) pour la régression sont utilisées.  

**Optimiseur:**  L'optimiseur est l'algorithme utilisé pour ajuster les poids du réseau pendant la rétropropagation afin de minimiser la fonction de perte. Des optimiseurs populaires incluent Adam, SGD (Stochastic Gradient Descent), RMSProp, etc. Chacun a ses avantages en termes de vitesse de convergence, gestion du bruit, et adaptation du taux d'apprentissage.  

**Taux d'apprentissage (Learning Rate):** Il contrôle la taille des pas effectués lors de l'optimisation des poids du réseau. Un taux d'apprentissage trop élevé peut entraîner une divergence, tandis qu'un taux trop faible peut ralentir la convergence. Le réglage optimal du taux d'apprentissage est crucial pour un apprentissage efficace du modèle.  

**Batch Size:** Il définit le nombre d'échantillons utilisés pour estimer le gradient des poids du réseau avant de mettre à jour ces poids. Un batch size plus grand peut accélérer le processus d'apprentissage en exploitant l'efficacité des calculs matriciels, tandis qu'un batch size plus petit peut fournir une estimation de gradient plus fréquente et potentiellement plus précise.  

**Nombre de couches cachées et de neurones par couche:** Le nombre de couches cachées et le nombre de neurones par couche déterminent la complexité et la capacité de représentation du réseau. Plus il y a de couches et de neurones, plus le modèle peut apprendre des représentations complexes des données. Cependant, trop de couches ou de neurones peuvent conduire à un surapprentissage (overfitting).  

<u>**Fonctionnement du MLP**</u>  
Le Perceptron Multicouches fonctionne en deux phases principales :  
- la propagation avant (forward propagation) pour calculer les prédictions  
- la rétropropagation (backpropagation) pour ajuster les poids du réseau en fonction de la fonction de perte.  

**Forward Propagation:**  
Lorsque des données sont introduites dans le réseau, elles traversent les différentes couches en commençant par les couches d'entrée. Chaque neurone dans une couche cachée calcule une combinaison linéaire des activations de la couche précédente, suivie de l'application d'une fonction d'activation non linéaire comme ReLU, sigmoid ou tanh. Cette transformation se répète à travers chaque couche cachée jusqu'à ce que les données atteignent la couche de sortie. Les neurones de la couche de sortie génèrent alors les prédictions du modèle en fonction des activations finales des neurones précédents.  

**Backpropagation:**  
Une fois les prédictions calculées, le réseau compare ces prédictions avec les valeurs réelles à l'aide d'une fonction de perte spécifique comme la cross-entropy pour la classification ou le MSE pour la régression. Cette fonction de perte mesure la différence entre les prédictions du modèle et les vérités terrain.
En utilisant l'algorithme de rétropropagation, cette erreur est propagée de la couche de sortie vers les couches cachées, ajustant progressivement les poids du réseau pour minimiser la fonction de perte. Cela se fait en calculant les gradients des poids par rapport à la fonction de perte à l'aide de la dérivation chainée (chain rule).  

<u>**Exemple d'application:**</u> 
*- Classification d'Images* :  
Dans un problème de classification d'images, un MLP pourrait prendre en entrée des images représentées par des pixels. Chaque neurone dans la couche d'entrée correspond à un pixel. Les couches cachées du MLP apprennent à extraire des caractéristiques hiérarchiques des images, comme les contours, les textures, et les formes.
La couche de sortie aurait un neurone par classe, activé par une fonction comme softmax pour prédire la probabilité de chaque classe.  

*- Régression* :  
Pour une tâche de régression, où l'objectif est de prédire une valeur numérique, un MLP pourrait avoir une couche de sortie avec un seul neurone activé linéairement pour prédire une valeur continue.
Les couches cachées permettent au réseau d'apprendre des relations complexes entre les variables d'entrée et la variable de sortie continue.  

En combinant propagation avant et rétropropagation, le Perceptron Multicouches est capable d'apprendre à partir des données et d'ajuster ses paramètres internes pour améliorer ses performances prédictives.

#### 2.1.2] Choix d'architecture du PMC  
Le choix de l'architecture du Perceptron Multicouches, en fonction de la problématique de classification ou de régression, est crucial pour obtenir des résultats optimaux et adaptés aux besoins spécifiques de chaque type de tâche. Voici comment l'architecture est adaptée à chacune de ces problématiques :

<u>**Pour la Classification**</u>
Dans le contexte de la classification, où l'objectif est de prédire une étiquette de classe pour chaque instance de données, l'architecture du PMC est généralement structurée de la manière suivante :

*Couches d'Entrée* : Cette première couche reçoit les caractéristiques ou les variables des données d'entrée. Le nombre de neurones dans cette couche est déterminé par le nombre de caractéristiques dans les données d'entrée.

*Couches Cachées* : Pour la classification, plusieurs couches cachées permettent au PMC d'apprendre des représentations hiérarchiques des données. Les choix typiques incluent l'utilisation de couches avec des fonctions d'activation non linéaires telles que ReLU, tanh, ou sigmoid. Plusieurs couches cachées permettent au modèle d'apprendre des caractéristiques abstraites et complexes des données, ce qui peut améliorer la capacité du modèle à discriminer entre différentes classes.

*Couche de Sortie* : La dernière couche du PMC dans un problème de classification est une couche de sortie composée de neurones correspondant au nombre de classes à prédire. La fonction d'activation utilisée dans cette couche est généralement une softmax pour obtenir des probabilités pour chaque classe. Cela permet de classifier chaque instance d'entrée dans l'une des classes prédéfinies.

<u>**Pour la Régression**</u>  
En revanche, dans le cas de la régression où l'objectif est de prédire une valeur continue plutôt qu'une classe, l'architecture du PMC est ajustée de manière légèrement différente :

*Couches d'Entrée* : Comme pour la classification, cette couche reçoit les caractéristiques des données d'entrée.

*Couches Cachées* : Les couches cachées sont également utilisées pour apprendre des représentations complexes des données. Cependant, le nombre de couches et le nombre de neurones peuvent être ajustés en fonction de la complexité des relations à modéliser dans les données de régression.

*Couche de Sortie* : Contrairement à la classification, la couche de sortie dans la régression est généralement composée d'un seul neurone qui produit une valeur continue en sortie. La fonction d'activation dans cette couche est souvent linéaire ou une fonction identité, car elle permet de prédire directement des valeurs numériques sans restriction de plage comme dans la classification.

En choisissant une architecture appropriée pour le Perceptron Multicouches en fonction de la classification ou de la régression, on peut maximiser la capacité prédictive du modèle tout en optimisant les ressources computationnelles et la complexité du réseau. Cette adaptation fine de l'architecture garantit que le modèle est bien adapté aux caractéristiques spécifiques des données et aux exigences particulières de chaque problème de machine learning.

#### 2.1.3] Définitions  
**1. Fonction d'activation :**  
Une fonction d'activation est une fonction mathématique appliquée à la sortie d'un neurone dans un réseau de neurones artificiels. Elle introduit de la non-linéarité dans le modèle, permettant ainsi au réseau de modéliser des relations complexes et non linéaires entre les variables d'entrée et de sortie.  
Voici quelques exemples de fonctions d'activation couramment utilisées :  

- ReLU (Rectified Linear Unit): ReLU($x$) = max(0, $x$)  
- Sigmoid: $\theta$($x$) = $\frac{1}{1+e^-x}$  
- Tanh (Tangente hyperbolique): tanh($x$) = $\frac {e^x - e^-x}{e^x + e^-x}$

**2. Forward Propagation :**  
La propagation avant est le processus par lequel les données sont introduites dans un réseau de neurones et propagées à travers les différentes couches du réseau, de la couche d'entrée à la couche de sortie. Chaque couche calcule une combinaison linéaire des entrées pondérées par les poids associés à chaque neurone, suivie de l'application d'une fonction d'activation.  

Exemple mathématique (avec ReLU) : Pour une couche cachée : z = W x $x$ + b, a = ReLU($z$)  

$x$: Vecteur d'entrée  
$W$: Matrice de poids  
$b$: Vecteur de biais  
$z$: Résultat de la combinaison linéaire  
$a$: Activation après application de ReLU  

**3. Retropropagation :**  
La rétropropagation est l'algorithme utilisé pour calculer le gradient de la fonction de perte par rapport aux poids du réseau, en partant de la couche de sortie et en remontant jusqu'à la couche d'entrée. Cela permet de mettre à jour les poids du réseau pour minimiser la fonction de perte, en utilisant des techniques d'optimisation telles que la descente de gradient.

Exemple conceptuel : Après avoir calculé l'erreur de prédiction à partir de la fonction de perte, la rétropropagation ajuste les poids du réseau pour minimiser cette erreur en propageant le gradient de l'erreur de la sortie vers les couches précédentes.  

**4. Loss function :**  
La fonction de perte mesure la différence entre les prédictions du modèle et les valeurs réelles attendues lors de l'entraînement d'un réseau de neurones. Son objectif est de quantifier la performance du modèle et de guider l'algorithme d'optimisation (comme la descente de gradient) pour ajuster les poids du réseau.  

*Exemple*:  
Pour une tâche de classification binaire, une fonction de perte couramment utilisée est la Cross-Entropy Loss, définie comme :  
Cross-Entropy Loss = 
$$
\sum_{i} y_i \log(p_i) + (1 - y_i) \log(1 - p_i)
$$

$y_{i}$ : Valeur réelle (0 ou 1)  
$p_{i}$ Probabilité prédite pour la classe positive  

**5. Descente de Gradient (Gradient Descent) :**  
La descente de gradient est une technique d'optimisation utilisée pour minimiser la fonction de perte en ajustant itérativement les poids du réseau de neurones. Elle fonctionne en calculant le gradient de la fonction de perte par rapport aux poids du réseau, puis en ajustant les poids dans la direction opposée du gradient.  

*Exemple*:  
Pour mettre à jour les poids $𝑊$ d'une couche cachée dans un réseau de neurones en utilisant la descente de gradient, on utilise la règle de mise à jour :  
$$
W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla_{W} L
$$

$W_{new}$: Nouveaux poids après la mise à jour   
$W_{old}$: Anciens poids  
$\eta$: taux d'apprentissage (learning rate), un hyperparamètre qui contrôle la taille des pas effectués lors de la descente de gradient.  
$\nabla_{W} L$: Gradient du loss function $L$ par rapport aux poids $W$, indiquant la direction et l'amplitude du changement à effectuer pour minimiser la perte $L$.  

**6. Vanishing Gradients**:  
Les gradients qui s'approchent de zéro de manière exponentielle à mesure que la profondeur du réseau augmente, posant ainsi des problèmes lors de l'entraînement.


#### 2.1.4] Fonction d'activation
Une fonction d’activation est un composant fondamental des réseaux de neurones artificiels, utilisé pour introduire la non-linéarité dans le modèle. En termes simples, elle transforme les signaux entrants d’un neurone pour déterminer si ce neurone doit être activé ou non, c’est-à-dire s’il doit transmettre les informations aux neurones suivants.

Dans un réseau de neurones, les signaux bruts, ou données d’entrée, sont pondérés et cumulés dans chaque neurone. La fonction d’activation prend ce cumul et le transforme en une sortie utilisable. Le terme ‘potentiel d’activation’ provient de l’équivalent biologique et représente le seuil de stimulation qui déclenche une réponse du neurone. Ce concept est important dans les réseaux de neurones artificiels car il permet de déterminer quand un neurone doit être activé, en fonction de la somme pondérée des entrées.

Sans fonction d’activation, le modèle serait simplement une combinaison linéaire d’entrées, incapable de résoudre des problèmes complexes. En introduisant la non-linéarité, les fonctions d’activation permettent au réseau de neurones de modéliser des relations complexes et d’apprendre des représentations abstraites des données.

Il existe plusieurs types de fonctions d’activation, chacune avec des caractéristiques et des applications spécifiques, comme : 
- la fonction Sigmoid
- la fonction Tanh (Tangente Hyperbolique) 
- la fonction ReLU (Rectified Linear Unit). 

Ces fonctions sont choisies en fonction des besoins spécifiques du modèle et des données avec lesquelles il travaille.
‍
#### 2.1.5] Définition & différenciation d'hyperparamètres
##### Qu'est-ce qu'une Epoch ?
Un cycle complet du jeu de données d’entraînement est considéré comme une « époque » dans le domaine du Machine Learning. Elle reflète le nombre de passages de l’algorithme au cours de la phase d’entraînement.

On peut définir une epoch comme le nombre de passages d’un dataset d’entraînement par un algorithme. Un passage équivaut à un aller-retour.

Le nombre d’epochs peut atteindre plusieurs milliers, car la procédure se répète indéfiniment jusqu’à ce que le taux d’erreurs du modèle soit suffisamment réduit.

Une époque est composée d’une agrégation de « batches » ou « lots » de données et d’itérations. Les jeux de données sont généralement décomposés en batches, tout particulièrement lorsque le volume de données est massif.

##### Qu'est-ce qu'une itération ?
Dans le domaine du Machine Learning, une itération indique le nombre de fois que les paramètres d’un algorithme sont modifiés. Les implications spécifiques dépendent du contexte.

En général, une itération d’entraînement d’un réseau de neurones inclut le « batch processing » ou traitement de lot du dataset, le calcul de la fonction de coût, la modification et la rétropropagation de tous les facteurs de poids.

L’itération et l’époque sont souvent confondues à tort. En réalité, une itération implique le traitement d’un batch tandis qu’une époque désigne le traitement de toutes les données du dataset.

Par exemple, si une itération traite 10 images d’un ensemble de 1000 images avec une taille de batch de 10, il faudra 100 itérations pour terminer une époque.

##### Qu'est-ce qu'un batch ?
Les données d’entraînement sont décomposées en plusieurs petits « lots » ou « batches » en anglais. Le but est d’éviter les problèmes liés à un manque d’espace de stockage.

Les batches peuvent être facilement utilisés pour nourrir le modèle de Machine Learning afin de l’entraîner. Ce processus de décomposition du dataset est appelé « batch ».

Une epoch peut être composée d’un batch ou davantage. Le nombre d’échantillons d’entraînement utilisés lors d’une itération est la « taille de lot » ou « batch size ». On distingue trois possibilités.

Dans le cas du « Batch Mode », les valeurs d’itération et d’époque sont égales puisque la taille du batch est égale au dataset complet. Une itération équivaut donc à une époque.

En « mini-batch mode », la taille du dataset complet est inférieure à la taille de batch. Par conséquent un seul batch est plus large que le jeu de données d’entraînement.

Enfin, en mode stochastique, la taille du batch est unique. Par conséquent, le gradient et les paramètres du réseau de neurones sont changés à chaque échantillon.

#### 2.1.6] Learning Rate
Dans le domaine de l’intelligence artificielle, le taux d’apprentissage (learning rate) est le facteur multiplicatif appliqué au gradient. À chaque itération, l'algorithme de descente de gradient multiplie le taux d'apprentissage par le gradient.

Le taux d'apprentissage est un hyperparamètre qui joue sur la rapidité de la descente de gradient : un nombre d’itérations plus ou moins important est nécessaire avant que l’algorithme ne converge, c’est-à-dire qu’un apprentissage optimal du réseau soit réalisé.

Lorsque le learning rate est trop petit, le modèle apprend trop lentement, il peut mettre des heures à converger, tandis que lorsqu'il est trop grand, le modèle fait de grands sauts et ne peut jamais converger, ou diverger (devenir instable).

#### 2.1.7] Batch Normalization
La batch normalization est un procédé utilisé en deep learning pour améliorer la performance de réseaux de neurone. Elle permet de normaliser les sorties à chaque couche du réseau en ajustant la moyenne et l'écart type puis les utilise pour standardiser les données.

#### 2.1.8] Algorithme d'Optimisation d'Adam
Adam est un optimiseur utilisé pour réduire le temps d'entrainement de grand modèles. il utilise des paramètres adaptatifs  pour chaque modèle ce qui permet de conduire à une convergence plus rapide.

#### 2.1.9] Définir simplement le Perceptron Multicouches
le perceptron multicouche est un type de réseau de neurones artificiel organisé en plusieurs couches

### 2.2] Convolutional Neural Networs (CNN)

#### 2.2.1] Réseaux de neurones artificiels de type convolutif
L’architecture d’un réseau de neurones convolutifs est construite pour extraire des valeurs pertinentes à partir de données visuelles complexes. Une capacité rendue possible par l’intégration de layers ou couches fondamentales dans la structure du réseau : Convolutional, Pooling et Fully-Connected.

 ##### Hyperparamètres typiques d'un CNN
- Taille des filtres : 3×3 ou 5×5 (zone de détection locale)
- Nombre de filtres : 32, 64, 128… (capacité d’extraction de caractéristiques)
- Stride : 1 ou 2 (pas du filtre)
- Padding : valid ou same (garde ou réduit la taille des cartes)
- Pooling : 2×2 (réduction de dimension, souvent max pooling)
- Fonction d’activation : ReLU, LeakyReLU (non-linéarité)
- Taille des couches denses : 128, 256… (pour la classification)
- Taux d’apprentissage : 0.001 typiquement (vitesse d’apprentissage)
- Batch size : 32, 64… (taille des lots)
- Nombre d’époques : 10 à 100+ (nombre de passages sur les données)
- Dropout : 0.2 à 0.5 (régularisation)
- Optimiseur : Adam, SGD, RMSprop (mise à jour des poids)


#### 2.2.2] Couche convolutive
La couche de convolution est conçue pour extraire des caractéristiques significatives des données en effectuant une opération de convolution. Des filtres ou noyaux sont appliqués sur l’ensemble des entrées. Chaque filtre, représenté par une matrice de poids, est appliqué de manière glissante sur l’image ou les données d’entrée. Ils calculent la somme pondérée des valeurs à chaque position. Des cartes de caractéristiques sont ainsi générées, mettant en évidence des motifs locaux ou des structures marquantes dans la data. 

La couche de convolution est donc un élément essentiel. Elle permet la reconnaissance des motifs ou pattern par le réseau de neurones, en partageant les poids du filtre à travers différentes parties de l’entrée. Après la convolution, on retrouve une fonction d’activation, généralement ReLU (Rectified Linear Unit). Elle introduit la notion de non-linéarité dans le modèle.

#### 2.2.3] Quelle fonction d'activation pour CNN
es fonctions d'activation les plus couramment utilisées dans les CNN sont : Unité linéaire rectifiée (ReLU) Sigmoïde Tangente hyperbolique (tanh)

#### 2.2.4] Feature Map
Une feature map est la carte qui permet d'obtenir la localisation des caractéristiques dans l'image.

#### 2.2.5] Couche de Pooling
La couche de pooling est une opération appliquée entre deux couches de convolution. Elle reçoit en entrée les features map produites en sortie par une couche de convolution et est chargée de réduire la taille des images tout en maintenant les caractéristiques les plus essentielles.

On a le max-pooling et le average pooling

#### 2.2.6] Couche connectée
Les couches FC sont placées en fin d'architecture et sont entièrement connectées à tous les neurones de sorties.
Après avoir reçu un vecteur en entrée, elle applique successivement une combinaison linéaire puis une fonction d'activation dans le but final de classifier l'input de l'image.

#### 2.2.7] Pourquoi préférer un Réseau de neurones convolutifs à un réseau dense pour une tâche de classification
les CNN sont adaptés pour traiter les données d'images en raison de leur structure convolutive qui permet  de reconnaitre les caractéristiques spécifiques des parties de l'image, tandis que les réseaux de neurones denses sont plus adaptés pour les tâches de classification où les données d'entrées sont déjà vectorielles.