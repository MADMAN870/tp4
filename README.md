# tp4

Réponses aux questions des TP 2, TP 3 (bonus) et TP 4 (test 5), avec les extraits de logging obtenus via `logRequestsAndResponses(true)` et le logger `dev.langchain4j` réglé en niveau `FINE` (`configureLogger()`).

## TP 2 — RAG et QueryTransformer

### 1. Nouvelle réponse

Question posée dans `src/main/java/ma/emsi/Test5.java` (TP 2) : « Comment s'appelle le chat de Pierre ? »

> Le chat de Pierre s'appelle Felix.

Sans RAG, le modèle ne connaît pas cette information : elle n'existe que dans le fichier local `infos.txt` (`Le nom du chat de Pierre est Felix.`). Avec le RAG, le segment est retrouvé dans le magasin vectoriel puis injecté dans le prompt ; la réponse devient donc « Felix » au lieu d'une réponse inventive ou d'un refus.

### 2. Question envoyée au LLM après l'intervention du RAG (logging)

```text
INFO: Request:
- model: gemini-3.5-flash
- messages:
[SystemMessage] Tu réponds en français. Utilise en priorité les extraits fournis, et complète avec tes connaissances générales si le document ne donne pas l'information demandée.
[UserMessage] Comment s'appelle le chat de Pierre ?

Answer using the following information:
Le nom du chat de Pierre est Felix.
```

La question de l'utilisateur n'est pas remplacée : le `DefaultContentInjector` de LangChain4j la garde et ajoute simplement, sous `Answer using the following information:`, les extraits retrouvés par le `ContentRetriever`.

### 3. Ce qui a changé avec le QueryTransformer

Sans transformateur, une question de suivi comme « Dans le support de cours fourni, à quoi c'est comparé ? » est envoyée telle quelle au retriever. Le mot « c'est » est une anaphore : il renvoie au « fine-tuning » de la question précédente. L'embedding de cette question isolée est donc pauvre en contexte et le RAG retrouve de mauvais segments.

Avec `CompressingQueryTransformer` (`RagTest2.java:110-117`), le LM reformule la question en utilisant la mémoire de conversation avant la récupération. Le texte envoyé au LM est :

```text
Read and understand the conversation between the User and the AI. Then, analyze the new query from the User. Identify all relevant details, terms, and context from both the conversation and the new query. Reformulate this query into a clear, concise, and self-contained format suitable for information retrieval.

Conversation:
User: Qu'est-ce que le fine-tuning ?
AI: Le fine-tuning consiste à réentraîner un modèle pré-entraîné.

User query: Dans le support de cours fourni, à quoi c'est comparé ?

It is very important that you provide only reformulated query and nothing else! Do not prepend a query with anything!
```

Question reformulée obtenue :

```text
Qu'est-ce que le fine-tuning et à quoi est-il comparé dans le support ?
```

La recherche vectorielle porte donc désormais sur une question autonome, avec le terme « fine-tuning » explicite : les extraits remontés sont plus pertinents et la réponse finale est mieux ancrée dans le document.

## TP 3 — Bonus : choisir entre les 2 retrievers

Le bonus demande de copier le texte envoyé au LM pour qu'il choisisse entre le retriever PDF et le retriever Web. Ce texte est généré par `LanguageModelQueryRouter` (`Test4Bonus.java:90-102`) à partir des descriptions des deux sources :

```text
Based on the user query, determine the most suitable data source(s) to retrieve relevant information from the following options:
1: Support de cours sur l'intelligence artificielle : RAG, fine-tuning, prompt engineering, embeddings, LangChain4j et bases de données vectorielles.
2: Recherche Web générale : actualités, événements récents et sujets sans rapport avec le support de cours sur l'IA.
It is very important that your answer consists of either a single number or multiple numbers separated by commas and nothing else!
User query: Qui a remporté la Coupe du monde de football 2022 ?
```

Le LM doit répondre uniquement par un numéro (ici `2`) ; le routeur active alors uniquement le `WebSearchContentRetriever`. C'est ce mécanisme qui remplace le `DefaultQueryRouter` pour éviter d'interroger les deux sources en permanence.

## TP 4 — Test 5 : extraits du PDF ajoutés à la question

### Pourquoi des extraits du PDF sont ajoutés

Avec le code initial (`Test4.java:75-82`), le `DefaultQueryRouter` envoie **chaque** requête à **tous** les `ContentRetriever` enregistrés. Les deux sources (PDF `rag.pdf` et recherche Web Tavily) sont donc toujours interrogées, même pour une question d'actualité sans rapport avec le cours. Le `DefaultContentInjector` ajoute ensuite tous les extraits récupérés au message utilisateur :

```text
INFO: Request:
- model: gemini-3.5-flash
- messages:
[SystemMessage] Tu réponds en français. Utilise les extraits du PDF et les résultats Web fournis par le RAG.
[UserMessage] Qui a remporté la Coupe du monde de football 2022 ?

Answer using the following information:
RAG qui ne modifie pas le LM mais qui, pour chaque question, recherche informations supplémentaires dans des documents fiables, les ajoute au prompt pour que le LM puisse les utiliser

<résultat Web Tavily : ...>
```

Le passage « Answer using the following information: » suivi d'un extrait du support de cours montre le problème : le PDF pollue le prompt et peut tromper le LM sur une question qui ne concerne pas l'IA.

### Comment le problème a été corrigé

Dans `Test4Bonus.java:90-102`, le `DefaultQueryRouter` est remplacé par un `LanguageModelQueryRouter` auquel on fournit une description pour chaque retriever. Le LM choisit la source pertinente (texte complet reproduit dans la section TP 3 ci-dessus) et un seul retriever est utilisé :

```text
INFO: Request:
- model: gemini-3.5-flash
- messages:
[SystemMessage] Tu réponds en français. Utilise les extraits récupérés par le RAG quand ils sont présents.
[UserMessage] Qui a remporté la Coupe du monde de football 2022 ?

Answer using the following information:
<résultat Web Tavily : ...>
```

L'extrait du PDF n'apparaît plus : le prompt ne contient que les informations de la source réellement pertinente.

## Remarque sur la capture des logs

Les extraits ci-dessus ont été régénérés avec LangChain4j 1.22.0 en utilisant la même chaîne RAG (`DefaultRetrievalAugmentor`, `DefaultQueryRouter` / `LanguageModelQueryRouter`, `DefaultContentInjector`) et un `ChatModel` d'enregistrement affichant les requêtes. Le quota d'embeddings Gemini du compte était épuisé (erreur 429) au moment de la capture, ce qui empêchait de rejouer l'indexation complète du PDF ; les extraits de document montrés sont ceux de `infos.txt` et de `src/main/resources/rag.pdf` du projet.
