package ma.emsi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    public static void main(String[] args) {
        configureLogger();

        String cle = System.getenv("GEMINI_KEY");

        if (cle == null || cle.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement GEMINI_KEY n'est pas définie."
            );
        }

        String modeleChat = System.getenv().getOrDefault(
                "GEMINI_CHAT_MODEL",
                "gemini-3.5-flash"
        );

        ChatModel chatModel = creerChatModel(cle, modeleChat);
        EmbeddingModel embeddingModel = creerEmbeddingModel(cle);

        Document documentIA = chargerDocument("rag.pdf");
        Document documentCuisine = chargerDocument("cuisine.txt");

        EmbeddingStore<TextSegment> storeIA =
                creerStore(documentIA, embeddingModel);
        EmbeddingStore<TextSegment> storeCuisine =
                creerStore(documentCuisine, embeddingModel);

        System.out.println("Segments IA : " + nombreSegments(documentIA));
        System.out.println("Segments cuisine : " + nombreSegments(documentCuisine));

        ContentRetriever retrieverIA = creerRetriever(storeIA, embeddingModel);
        ContentRetriever retrieverCuisine = creerRetriever(storeCuisine, embeddingModel);

        Map<ContentRetriever, String> descriptions = new LinkedHashMap<>();
        descriptions.put(
                retrieverIA,
                "Support de cours sur l'intelligence artificielle : modèles de langage, " +
                "prompt engineering, fine-tuning, RAG, embeddings, LangChain4j et bases de données vectorielles."
        );
        descriptions.put(
                retrieverCuisine,
                "Document de cuisine française : recettes, ingrédients et préparation " +
                "de plats comme la ratatouille, les crêpes et la salade de tomates au basilic."
        );

        QueryRouter routeur = new LanguageModelQueryRouter(chatModel, descriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(routeur)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .systemMessage(
                        "Tu réponds en français en t'appuyant sur les extraits récupérés " +
                        "par le RAG. Si les extraits ne contiennent pas la réponse, " +
                        "dis-le clairement."
                )
                .build();

        poserQuestion(assistant, "Qu'est-ce que le RAG ?");
        poserQuestion(assistant, "Comment préparer une ratatouille ?");
        poserQuestion(assistant, "Quelle est la différence entre le fine-tuning et le RAG ?");
        poserQuestion(assistant, "Quels sont les ingrédients des crêpes ?");
    }

    private static ChatModel creerChatModel(String cle, String modele) {
        return GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName(modele)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();
    }

    private static EmbeddingModel creerEmbeddingModel(String cle) {
        return GoogleGenAiEmbeddingModel.builder()
                .apiKey(cle)
                .modelName("gemini-embedding-001")
                .taskType(GoogleGenAiEmbeddingModel.TaskTypeEnum.RETRIEVAL_DOCUMENT)
                .timeout(Duration.ofSeconds(60))
                .build();
    }

    private static Document chargerDocument(String nomFichier) {
        return ClassPathDocumentLoader.loadDocument(
                nomFichier,
                new ApacheTikaDocumentParser()
        );
    }

    private static int nombreSegments(Document document) {
        DocumentSplitter splitter = DocumentSplitters.recursive(2000, 200);
        return splitter.split(document).size();
    }

    private static EmbeddingStore<TextSegment> creerStore(
            Document document,
            EmbeddingModel embeddingModel
    ) {
        DocumentSplitter splitter = DocumentSplitters.recursive(2000, 200);
        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        return store;
    }

    private static ContentRetriever creerRetriever(
            EmbeddingStore<TextSegment> store,
            EmbeddingModel embeddingModel
    ) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    private static void poserQuestion(Assistant assistant, String question) {
        System.out.println("\nQuestion : " + question);
        System.out.println("Réponse : " + assistant.chat(question));
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}
