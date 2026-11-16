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
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagTest2 {

    private static final String QUESTION_1 =
            "Qu'est-ce que le fine-tuning ?";

    private static final String QUESTION_2 =
            "Dans le support de cours fourni, à quoi c'est comparé ?";

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

        ChatModel chatModel = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName(modeleChat)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(2000, 200);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = GoogleGenAiEmbeddingModel.builder()
                .apiKey(cle)
                .modelName("gemini-embedding-001")
                .taskType(GoogleGenAiEmbeddingModel.TaskTypeEnum.RETRIEVAL_DOCUMENT)
                .timeout(Duration.ofSeconds(60))
                .build();

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        System.out.println("\n=== TEST 2 SANS QUERY TRANSFORMER ===");
        Assistant assistantSansTransformation = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .systemMessage(
                        "Tu réponds en français. Utilise en priorité les extraits fournis, " +
                        "et complète avec tes connaissances générales si le document " +
                        "ne donne pas l'information demandée."
                )
                .build();

        System.out.println("Q1 : " + QUESTION_1);
        System.out.println("R1 : " + assistantSansTransformation.chat(QUESTION_1));
        System.out.println("Q2 : " + QUESTION_2);
        System.out.println("R2 : " + assistantSansTransformation.chat(QUESTION_2));

        System.out.println("\n=== TEST 2 AVEC COMPRESSING QUERY TRANSFORMER ===");
        QueryTransformer queryTransformer = CompressingQueryTransformer.builder()
                .chatModel(chatModel)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(retriever)
                .queryTransformer(queryTransformer)
                .build();

        Assistant assistantAvecTransformation = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .systemMessage(
                        "Tu réponds en français. Utilise en priorité les extraits fournis, " +
                        "et complète avec tes connaissances générales si le document " +
                        "ne donne pas l'information demandée."
                )
                .build();

        System.out.println("Q1 : " + QUESTION_1);
        System.out.println("R1 : " + assistantAvecTransformation.chat(QUESTION_1));
        System.out.println("Q2 : " + QUESTION_2);
        System.out.println("R2 : " + assistantAvecTransformation.chat(QUESTION_2));
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}
