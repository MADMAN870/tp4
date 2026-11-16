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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3 {

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

        Document document = chargerDocument("rag.pdf");
        EmbeddingStore<TextSegment> store = creerStore(document, embeddingModel);
        ContentRetriever retriever = creerRetriever(store, embeddingModel);

        System.out.println("Segments RAG : " + segments(document).size());

        PromptTemplate templateRouteur = PromptTemplate.from(
                "Est-ce que la requête '{{question}}' porte sur l'IA ? " +
                "Réponds seulement par 'oui', 'non' ou 'peut-être'."
        );

        class RouteurIAOuPas implements QueryRouter {

            private final ChatModel modele;
            private final ContentRetriever contentRetriever;
            private final PromptTemplate promptTemplate;

            RouteurIAOuPas(
                    ChatModel modele,
                    ContentRetriever contentRetriever,
                    PromptTemplate promptTemplate
            ) {
                this.modele = modele;
                this.contentRetriever = contentRetriever;
                this.promptTemplate = promptTemplate;
            }

            @Override
            public Collection<ContentRetriever> route(Query query) {
                String prompt = promptTemplate
                        .apply(Map.of("question", query.text()))
                        .text();

                String decision = modele.chat(prompt)
                        .trim()
                        .toLowerCase(Locale.ROOT);

                System.out.println("Décision du routeur : " + decision);

                if (decision.startsWith("non")) {
                    return List.of();
                }

                return List.of(contentRetriever);
            }
        }

        QueryRouter routeur = new RouteurIAOuPas(
                chatModel,
                retriever,
                templateRouteur
        );

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(routeur)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .systemMessage(
                        "Tu réponds en français. Utilise les extraits fournis par le RAG " +
                        "quand ils sont présents, sinon réponds normalement."
                )
                .build();

        poserQuestion(assistant, "Bonjour");
        poserQuestion(assistant, "Qu'est-ce que le RAG ?");
        poserQuestion(assistant, "Comment préparer une ratatouille ?");
        poserQuestion(assistant, "Quelle est la différence entre le fine-tuning et le RAG ?");
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

    private static List<TextSegment> segments(Document document) {
        DocumentSplitter splitter = DocumentSplitters.recursive(2000, 200);
        return splitter.split(document);
    }

    private static EmbeddingStore<TextSegment> creerStore(
            Document document,
            EmbeddingModel embeddingModel
    ) {
        List<TextSegment> segments = segments(document);

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
