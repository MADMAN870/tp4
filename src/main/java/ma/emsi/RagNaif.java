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
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaif {

    public static void main(String[] args) {
        configureLogger();

        String cle = System.getenv("GEMINI_KEY");

        if (cle == null || cle.isBlank()) {
            throw new IllegalStateException(
                    "La variable d'environnement GEMINI_KEY n'est pas définie."
            );
        }

        ChatModel chatModel = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(2000, 200);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        EmbeddingModel embeddingModel = GoogleGenAiEmbeddingModel.builder()
                .apiKey(cle)
                .modelName("gemini-embedding-001")
                .taskType(GoogleGenAiEmbeddingModel.TaskTypeEnum.RETRIEVAL_DOCUMENT)
                .timeout(Duration.ofSeconds(60))
                .logRequests(true)
                .build();

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        afficherSegmentsEnregistres(segments);

        String questionInitiale =
                "Quelle est la signification de 'RAG' ; à quoi ça sert ?";

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        Query query = Query.from(questionInitiale);
        List<Content> contenusRetrouves = retriever.retrieve(query);
        afficherSegmentsRetrouves(contenusRetrouves);

        afficherScores(embeddingModel, embeddingStore, questionInitiale);

        MessageWindowChatMemory memory =
                MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .systemMessage(
                        "Tu réponds en français. Utilise en priorité les extraits fournis, " +
                        "et complète avec tes connaissances générales si le document " +
                        "ne donne pas l'information demandée."
                )
                .build();

        System.out.println("\nQuestion : " + questionInitiale);
        System.out.println("Réponse : " + assistant.chat(questionInitiale));

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("Posez votre question (ou tapez 'fin') :");
                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                if ("fin".equalsIgnoreCase(question.trim())) {
                    System.out.println("Fin de la conversation.");
                    break;
                }

                System.out.println("Assistant : " + assistant.chat(question));
            }
        }
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private static void afficherSegmentsEnregistres(List<TextSegment> segments) {
        System.out.println("\n=== Segments enregistrés dans le magasin ===");
        for (int i = 0; i < segments.size(); i++) {
            System.out.println("--- Segment " + (i + 1) + " ---");
            System.out.println(segments.get(i).text());
        }
    }

    private static void afficherSegmentsRetrouves(List<Content> contenus) {
        System.out.println("\n=== Segments retrouvés pour la question ===");
        for (Content content : contenus) {
            System.out.println("--- Segment retrouvé ---");
            System.out.println(content.textSegment().text());
        }
    }

    private static void afficherScores(
            EmbeddingModel embeddingModel,
            InMemoryEmbeddingStore<TextSegment> embeddingStore,
            String question
    ) {
        Embedding questionEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(questionEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();

        EmbeddingSearchResult<TextSegment> resultat = embeddingStore.search(request);

        System.out.println("\n=== Scores des segments retrouvés ===");
        for (EmbeddingMatch<TextSegment> match : resultat.matches()) {
            System.out.println("Score : " + match.score());
            System.out.println(match.embedded().text());
        }
    }
}
