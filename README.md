# Construction un modèle de scoring pour prédire automatiquement la probabilité de faillite d'un client

### Contetxte

Une société financière propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre un outil de "scoring crédit" pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. 
Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit.

L'entreprise décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente
possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

### Objectifs

* Construire un modèle de scoring qui donnera automatiquement une prédiction sur la probabilité de faillite d'un client ;
* Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

### Spécifications du dashboard
* Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science ;
* Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre) ;
* Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.
  
### Les données

Les donnée suivantes ont été utilisées pour réaliser le dashboard (https://www.kaggle.com/c/home-credit-default-risk/data).


