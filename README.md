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

#### 2.1.5] Définition & différenciation d'hyperparamètres

#### 2.1.6] Learning Rate

#### 2.1.7] Batch Normalization

#### 2.1.8] Algorithme d'Optimisation d'Adam

#### 2.1.9] Définir simplement le Perceptron Multicouches

### 2.2] Convolutional Neural Networs (CNN)

#### 2.2.1] Réseaux de neurones artificiels de type convolutif

#### 2.2.2] Couche convolutive

#### 2.2.3] Quelle fonction d'activation pour CNN

#### 2.2.4] Feature Map

#### 2.2.5] Couche de Pooling

#### 2.2.6] Couche connectée

#### 2.2.7] Pourquoi préférer un Réseau de neurones convolutifs à un réseau dense pour une tâche de classification