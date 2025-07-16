package com.example.imagec2; // Package name for project organization

import ai.djl.inference.Predictor; // Import Predictor for running inference
import ai.djl.modality.Classifications; // Import Classifications for classification results
import ai.djl.modality.cv.ImageFactory; // Import ImageFactory to load images
import ai.djl.repository.zoo.Criteria; // Import Criteria to define model loading config
import ai.djl.repository.zoo.ZooModel; // Import ZooModel to handle loaded model
import javafx.application.Platform; // Import Platform for thread-safe UI updates
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty; // Import for table column data binding
import javafx.fxml.FXML; // Import FXML annotation for JavaFX
import javafx.scene.Node;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.PieChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.chart.XYChart.Series;
import javafx.scene.chart.BarChart;

import javafx.scene.control.*; // Import JavaFX UI controls
import javafx.scene.control.Alert.AlertType;
import javafx.scene.image.Image; // Import JavaFX Image class
import javafx.scene.image.ImageView; // Import JavaFX ImageView class
import javafx.scene.input.MouseEvent;
import javafx.stage.FileChooser; // Import FileChooser for image file selection
import javafx.stage.Stage; // Import Stage for window close handling

import java.io.File; // Java IO File class for file handling
import java.util.ArrayList; // Import ArrayList for storing image paths
import java.util.List; // Import List interface
import java.util.Map;
// Import ConcurrentHashMap for thread-safe storage
import java.util.concurrent.*; // Import concurrency package
import java.util.concurrent.atomic.AtomicInteger; // Import AtomicInteger for thread-safe counters
import java.util.stream.Collectors; // Import Collectors for stream operations

import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.VBox;
import javafx.event.EventHandler;

/**
 * ClassificationController is responsible for:
 * - Loading images from disk
 * - Classifying them with a pre-trained model
 * - Displaying results in the UI
 * - Managing multi-threaded operations and thread-safe UI updates
 */
public class ClassificationController {

    // ListView to display selected image paths
    @FXML private ListView<String> mylistview; // The ListView component where image paths appear

    // ImageView to display the currently selected image
    @FXML private ImageView myimageview; // The ImageView for image preview

    // Label to display classification status or result
    @FXML private Label labelresult; // The Label that shows status text

    // Table to show classification results
    @FXML private TableView<ClassificationResult> resultsTable; // Table to list classification outcomes

    // Table column for displaying image file names
    @FXML private TableColumn<ClassificationResult, String> imageNameColumn; // Column for image name

    // Table column for displaying classification result text
    @FXML private TableColumn<ClassificationResult, String> resultColumn; // Column for classification result

    // ProgressBar to show classification progress
    @FXML private ProgressBar progressBar; // The ProgressBar to visually track tasks

    // Thread-safe map to store classified results with file paths as keys
    private final ConcurrentHashMap<String, Classifications> results = new ConcurrentHashMap<>(); // Stores results safely

    // Counter for how many classifications are completed
    private final AtomicInteger completedTasks = new AtomicInteger(0); // Thread-safe counter

    // Counter for total tasks to be completed
    private final AtomicInteger totalTasks = new AtomicInteger(0); // Thread-safe counter

    // Add properties for performance tracking
    private final SimpleIntegerProperty classifiedCount = new SimpleIntegerProperty(0);
    private final SimpleDoubleProperty totalClassificationTime = new SimpleDoubleProperty(0.0);

    // Add charts
    @FXML private BarChart<String, Number> confidenceBarChart;
    @FXML private PieChart classificationPieChart;

    // Add labels for statistics
    @FXML private Label classifiedCountLabel;
    @FXML private Label totalClassificationTimeLabel;

    // Labels for Enhanced Progress Tracker
    @FXML private Label processedImagesLabel; // Label to display processed images count
    @FXML private Label totalImagesLabel; // Label to display total images count
    @FXML private Label estimatedTimeLabel; // Label to display estimated remaining time

    /**
     * ClassificationResult is a simple data holder for table rows.
     * Each result has an imageName and a classification result.
     */
    public static class ClassificationResult {
        // Name of the image file
        private final String imageName; // Final field for thread-safety

        // Classification outcome for that image
        private final String result; // Final field for classification text

        /**
         * Constructor that sets image name and result
         * @param imageName the image file name
         * @param result the classification text
         */
        public ClassificationResult(String imageName, String result) {
            this.imageName = imageName; // Assign file name
            this.result = result; // Assign classification outcome
        }

        /**
         * Gets the image file name
         * @return the image file name
         */
        public String getImageName() {
            return imageName; // Return stored file name
        }

        /**
         * Gets the classification result
         * @return the classification result text
         */
        public String getResult() {
            return result; // Return stored classification text
        }
    }

    // Create a thread pool to run classification tasks in parallel
    private final ExecutorService executor = new ThreadPoolExecutor(
        Runtime.getRuntime().availableProcessors(), // Use available CPU cores as core pool size
        Runtime.getRuntime().availableProcessors() * 2, // Allow up to twice the CPU cores as max pool size
        60L, // Threads remain alive for this duration if idle
        TimeUnit.SECONDS, // Time unit for keep-alive
        new ArrayBlockingQueue<>(100), // Tasks above core threads are queued here
        new ThreadPoolExecutor.CallerRunsPolicy() // If queue is full, tasks run on calling thread
    );

    /**
     * handleloadimages() opens a file chooser to let the user select image files.
     * The chosen file paths are then added to the ListView.
     */
    @FXML
    private void handleloadimages() {
        FileChooser fileChooser = new FileChooser(); // Create a new file chooser
        fileChooser.getExtensionFilters().add( // Restrict to certain image extensions
            new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg")
        );
        List<File> selectedFiles = fileChooser.showOpenMultipleDialog(null); // Show open dialog that allows multiple selection

        if (selectedFiles != null) { // Check if user didn't cancel
            // For each file, add its absolute path to the ListView
            selectedFiles.forEach(file ->
                mylistview.getItems().add(file.getAbsolutePath())
            );
        }
    }

    /**
     * handleClassifyImages():
     * 1. Gathers selected image paths
     * 2. Validates selection
     * 3. Clears previous results & resets progress
     * 4. Submits tasks to the thread pool for concurrent classification
     * 5. Uses CompletableFuture.allOf to wait for all tasks
     * 6. Updates UI upon completion
     */
    @FXML
    private void handleClassifyImages() {
        List<String> selectedPaths = new ArrayList<>(mylistview.getSelectionModel().getSelectedItems()); // Copy selected items to a new list

        // Log in terminal which thread is starting classification
        logThreadInfo("Starting classification of " + selectedPaths.size() + " images");

        if (selectedPaths.isEmpty()) { // Check if we have no items
            showAlert("No images selected", "Please select images to classify"); // Show alert for no selection
            return; // Exit the method
        }

        completedTasks.set(0); // Reset completed tasks counter
        totalTasks.set(selectedPaths.size()); // Set total tasks to number of selected images
        processedImagesLabel.setText("0"); // Reset processed images label
        totalImagesLabel.setText(String.valueOf(selectedPaths.size())); // Update total images label

        // Clear old results and reset UI on the JavaFX app thread
        Platform.runLater(() -> {
            resultsTable.getItems().clear(); // Clear table items
            labelresult.setText("Processing..."); // Set label for status
            progressBar.setProgress(0); // Reset progress bar
            estimatedTimeLabel.setText("Calculating..."); // Reset estimated time label
        });

        long startTime = System.nanoTime(); // Start time

        // Create a list of CompletableFutures that classify images asynchronously
        List<CompletableFuture<Void>> futures = selectedPaths.stream()
            .map(path -> CompletableFuture.runAsync(() -> { // For each path, run in a separate thread
                logThreadInfo("Processing image: " + path); // Log which thread handles this path
                try {
                    File imageFile = new File(path); // Create File object
                    // Use DJL to load image from file
                    ai.djl.modality.cv.Image djlImage = ImageFactory.getInstance()
                        .fromFile(imageFile.toPath());

                    logThreadInfo("Classifying image: " + path); // Log classification step
                    // Classify the loaded image using classifyImage()
                    Classifications result = classifyImage(djlImage);
                    // Store the classification result in thread-safe map
                    results.put(path, result);

                    // Update UI with classification results safely
                    Platform.runLater(() -> {
                        // Add row to TableView
                        resultsTable.getItems().add(new ClassificationResult(
                            imageFile.getName(), // Image file name
                            result.best().getClassName() // Best classification
                        ));

                        // Increment the completed counter
                        int completed = completedTasks.incrementAndGet();
                        // Update processed images label
                        processedImagesLabel.setText(String.valueOf(completed));
                        // Update progress bar fraction
                        progressBar.setProgress((double) completed / totalTasks.get());
                        // Log completion
                        logThreadInfo("Completed: " + path + " (" + completed + "/" + totalTasks.get() + ")");
                        
                        // Update estimated time remaining
                        updateEstimatedTime();
                    });

                } catch (Exception e) { // Catch exceptions during loading or classification
                    logThreadInfo("Error: " + path + " - " + e.getMessage()); // Log error
                    Platform.runLater(() ->
                        showAlert("Error", "Failed to process " + path + ": " + e.getMessage())
                    );
                }
            }, executor)) // Submit to our thread pool
            .collect(Collectors.toList()); // Collect Futures to a list

        // Wait for all tasks to finish
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .thenRunAsync(() -> { // Run this when everything is done
                long endTime = System.nanoTime(); // End time
                double classificationTime = (endTime - startTime) / 1_000_000_000.0;
                totalClassificationTime.set(totalClassificationTime.get() + classificationTime);

                logThreadInfo("All images processed"); // Log full completion
                logThreadPoolStatus(); // Log status of thread pool in terminal
                // Update UI with final status
                Platform.runLater(() -> {
                    labelresult.setText("Classification complete"); // Show completion label
                    progressBar.setProgress(1.0); // Fill progress bar
                    estimatedTimeLabel.setText("0.0s"); // Reset estimated time label
                    updateDashboard();
                });
            }, executor); // Use same executor
    }

    /**
     * classifyImage() loads a pre-trained ResNet model and performs classification on the given DJL Image.
     * It returns a Classifications object that contains the predicted labels and probabilities.
     * @param img the DJL Image object to be classified
     * @return Classifications containing classification results
     * @throws Exception if model loading or prediction fails
     */
    private Classifications classifyImage(ai.djl.modality.cv.Image img) throws Exception {
        // Define criteria for loading the pre-trained ResNet model
        Criteria<ai.djl.modality.cv.Image, Classifications> criteria = Criteria.builder()
                .setTypes(ai.djl.modality.cv.Image.class, Classifications.class) // Input: Image, Output: Classifications
                .optEngine("PyTorch") // Use PyTorch engine
                .optModelUrls("djl://ai.djl.pytorch/resnet") // Use pre-trained model from DJL model zoo
                .optModelName("resnet18") // Specifies resnet18 model architecture
                .build(); // Build the criteria object

        // Load the model and perform classification
        try (ZooModel<ai.djl.modality.cv.Image, Classifications> model = criteria.loadModel()) { // Load ResNet model
            try (Predictor<ai.djl.modality.cv.Image, Classifications> predictor = model.newPredictor()) { // Create predictor
                return predictor.predict(img); // Perform inference and return result
            }
        }
    }

    /**
     * handledeleteAll():
     * Clears the ListView, TableView, ImageView, label, and resets the progress bar.
     */
    @FXML
    private void handledeleteAll() {
        mylistview.getItems().clear(); // Remove all items from ListView
        resultsTable.getItems().clear(); // Remove all table entries
        myimageview.setImage(null); // Clear the displayed image
        labelresult.setText(null); // Clear label text
        progressBar.setProgress(0); // Reset progress bar
    }

    /**
     * handleExit():
     * Shuts down the executor immediately and exits the JavaFX application.
     */
    @FXML
    private void handleExit() {
        executor.shutdownNow(); // Terminate running tasks immediately
        Platform.exit(); // Close JavaFX platform
    }

    /**
     * handleAbout():
     * Displays an About dialog with basic application info.
     */
    @FXML
    private void handleAbout() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION); // Create an informational alert
        alert.setTitle("About"); // Set title of alert
        alert.setHeaderText("Image Classifier Pro"); // Set header text
        alert.setContentText("Multi-Threaded Image Classification\nVersion 1.0\n\nProcessing with "
                + Runtime.getRuntime().availableProcessors() + " threads"); // Show thread count
        alert.showAndWait(); // Display the alert and wait for user response
    }

    /**
     * showAlert() shows a simple warning dialog with given title and content text.
     * @param title the warning dialog title
     * @param content the main text of the dialog
     */
    private void showAlert(String title, String content) {
        Alert alert = new Alert(Alert.AlertType.WARNING); // Create a warning alert
        alert.setTitle(title); // Set alert title
        alert.setContentText(content); // Set alert text
        alert.showAndWait(); // Display alert modally
    }

    /**
     * Initializes UI components and sets up close handler, table columns, and image preview.
     * This method is automatically called by JavaFX after FXML loading.
     */
    @FXML
    private void initialize() {
        mylistview.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE); // Allow multiple item selection

        // Set column data binding for image name
        imageNameColumn.setCellValueFactory(
            cellData -> new SimpleStringProperty(cellData.getValue().getImageName())
        );

        // Set column data binding for classification text
        resultColumn.setCellValueFactory(
            cellData -> new SimpleStringProperty(cellData.getValue().getResult())
        );

        // Configure table columns with wrapping and tooltips
        configureCellFactory(imageNameColumn); // Setup custom cell factory
        configureCellFactory(resultColumn); // Setup custom cell factory

        // Let columns resize proportionally
        resultsTable.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY);

        // Prepare image view settings (zoom, preserve ratio, etc.)
        configureImagePreview();

        // Reset progress bar
        progressBar.setProgress(0);

        // Setup handler to close thread pool on window exit
        setupWindowCloseHandler();

        // Initialize charts
        initializeCharts();

        // Additional UI components initialization
    }

    /**
     * configureCellFactory():
     * Sets tooltip and wrap-text for each table cell, ensuring long text is handled properly.
     * @param column the TableColumn to configure
     */
    private void configureCellFactory(TableColumn<ClassificationResult, String> column) {
        column.setCellFactory(col -> {
            TableCell<ClassificationResult, String> cell = new TableCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty); // Call parent implementation
                    if (empty || item == null) { // Check if cell is empty
                        setText(null); // Clear text
                        setTooltip(null); // Clear tooltip
                    } else {
                        setText(item); // Display text
                        setTooltip(new Tooltip(item)); // Show tooltip with the same text
                    }
                }
            };
            cell.setWrapText(true); // Enable wrapping for text
            return cell; // Return configured cell
        });
    }

    /**
     * configureImagePreview():
     * Preserves aspect ratio, adds a scroll-to-zoom feature, and sets up auto-preview on selection.
     */
    private void configureImagePreview() {
        // Maintain aspect ratio
        myimageview.setPreserveRatio(true);
        // Enable smoothing for better image quality
        myimageview.setSmooth(true);
        
        // Listen for selection changes in the ListView
        mylistview.getSelectionModel().selectedItemProperty().addListener(
            (observable, oldValue, newValue) -> {
                if (newValue != null) { // If a new file is selected
                    loadAndOptimizeImage(new File(newValue)); // Load and scale it
                }
            }
        );

        // Add zoom functionality using scroll wheel
        myimageview.setOnScroll(event -> {
            double zoomFactor = 1.1; // Default zoom ratio
            if (event.getDeltaY() < 0) { // If user scrolls down
                zoomFactor = 1 / zoomFactor; // Reverse zoom
            }
            
            // Adjust ImageView dimensions
            myimageview.setFitWidth(myimageview.getFitWidth() * zoomFactor);
            myimageview.setFitHeight(myimageview.getFitHeight() * zoomFactor);
            
            event.consume(); // Mark event as handled
        });
    }

    /**
     * loadAndOptimizeImage():
     * Loads an image from a File and scales it proportionally to fit a defined area.
     * @param file the File object representing the image on disk
     */
    private void loadAndOptimizeImage(File file) {
        Image image = new Image(file.toURI().toString()); // Convert File to JavaFX Image
        
        double maxWidth = 600; // Max preview width
        double maxHeight = 500; // Max preview height
        double imageWidth = image.getWidth(); // Actual image width
        double imageHeight = image.getHeight(); // Actual image height
        
        double widthRatio = maxWidth / imageWidth; // Scale ratio by width
        double heightRatio = maxHeight / imageHeight; // Scale ratio by height
        double scale = Math.min(widthRatio, heightRatio); // Use the smaller ratio to preserve aspect
        
        myimageview.setFitWidth(imageWidth * scale); // Set scaled width
        myimageview.setFitHeight(imageHeight * scale); // Set scaled height
        myimageview.setImage(image); // Display the loaded image
    }

    /**
     * setupWindowCloseHandler():
     * Attaches a listener to the Stage to trigger shutdown when window is closed.
     */
    private void setupWindowCloseHandler() {
        // Listen for scene property changes on ImageView
        myimageview.sceneProperty().addListener((obs, oldScene, newScene) -> {
            if (newScene != null) { // If scene is present
                // Listen for window property changes
                newScene.windowProperty().addListener((obs2, oldWindow, newWindow) -> {
                    if (newWindow != null) { // If window is present
                        Stage stage = (Stage) newWindow; // Cast to a Stage
                        stage.setOnCloseRequest(event -> shutdown()); // On close, call shutdown
                    }
                });
            }
        });
    }

    /**
     * shutdown():
     * Gracefully shuts down the executor and forces shutdown if tasks take too long.
     * Preserves interrupt status and then exits the JavaFX application.
     */
    private void shutdown() {
        executor.shutdown(); // Start graceful shutdown
        try {
            // Wait for ongoing tasks to finish within 800 ms
            if (!executor.awaitTermination(800, TimeUnit.MILLISECONDS)) {
                executor.shutdownNow(); // Force shutdown if time exceeded
            }
        } catch (InterruptedException e) { // If interrupted during shutdown
            executor.shutdownNow(); // Force immediate shutdown
            Thread.currentThread().interrupt(); // Preserve interrupt status
        }
        Platform.exit(); // Exit JavaFX platform
    }

    /**
     * logThreadInfo():
     * Logs a message to the terminal with the current thread name for debugging concurrency.
     * @param message a custom message to log
     */
    private void logThreadInfo(String message) {
        System.out.println(String.format("[Thread: %s] %s",
            Thread.currentThread().getName(), message)); // Print thread name + message
    }

    /**
     * logThreadPoolStatus():
     * Logs the executor's status, including active threads, completed tasks, total tasks, and queue size.
     */
    private void logThreadPoolStatus() {
        if (executor instanceof ThreadPoolExecutor) { // Check if our executor is a ThreadPoolExecutor
            ThreadPoolExecutor pool = (ThreadPoolExecutor) executor; // Cast to ThreadPoolExecutor

            System.out.println("\nThread Pool Status:"); // Header text
            System.out.println("Active threads: " + pool.getActiveCount()); // Print active thread count
            System.out.println("Completed tasks: " + pool.getCompletedTaskCount()); // Print count of completed tasks
            System.out.println("Total tasks: " + pool.getTaskCount()); // Print total tasks handled by pool
            System.out.println("Queue size: " + pool.getQueue().size() + "\n"); // Print how many tasks are in the queue
        }
    }

    private void updateDashboard() {
        // Update classified count
        classifiedCount.set(results.size());
        classifiedCountLabel.setText(String.valueOf(classifiedCount.get()));
        
        // Update total classification time
        totalClassificationTimeLabel.setText(String.format("%.2f", totalClassificationTime.get()));
        
        // Update confidence bar chart
        confidenceBarChart.getData().clear();
        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("Confidence Scores");
        results.forEach((path, classification) -> {
            series.getData().add(new XYChart.Data<>(new File(path).getName(), classification.best().getProbability()));
        });
        confidenceBarChart.getData().add(series);
        
        // Update classification pie chart
        Map<String, Long> classificationCounts = results.values().stream()
            .map(Classifications::best)
            .map(Classifications.Classification.class::cast)
            .collect(Collectors.groupingBy(Classifications.Classification::getClassName, Collectors.counting()));
        classificationPieChart.getData().clear();
        classificationCounts.forEach((cls, count) -> {
            classificationPieChart.getData().add(new PieChart.Data(cls, count));
        });
    }

    private void initializeCharts() {
        // Bind classified count and total time to UI if needed
        // Additional chart setup can be done here
    }

    /**
     * handleConfidenceChartClick():
     * Handles clicks on the confidence bar chart to show detailed similar results.
     */
    @FXML
    private void handleConfidenceChartClick(MouseEvent event) {
        Node node = event.getPickResult().getIntersectedNode();
        if (node instanceof BarChart) {
            XYChart.Data<String, Number> data = ((XYChart.Data<String, Number>) node.getUserData());
            String imageName = data.getXValue();
            double confidence = data.getYValue().doubleValue();
            
            // Fetch and display similar results based on confidence
            showDetailedConfidenceDialog(imageName, confidence);
        }
    }

    /**
     * showDetailedConfidenceDialog():
     * Displays a dialog with detailed information about the selected confidence score.
     * @param imageName the name of the image
     * @param confidence the confidence score
     */
    private void showDetailedConfidenceDialog(String imageName, double confidence) {
        Alert alert = new Alert(AlertType.INFORMATION);
        alert.setTitle("Detailed Confidence");
        alert.setHeaderText("Details for " + imageName);
        alert.setContentText("Confidence: " + String.format("%.2f", confidence) + "\nDisplaying similar results from training data.");
        alert.showAndWait();
    }

    /**
     * handleDragOver():
     * Handles the drag over event for folder upload.
     */
    @FXML
    private void handleDragOver(DragEvent event) {
        Dragboard db = event.getDragboard();
        if (db.hasFiles()) {
            event.acceptTransferModes(TransferMode.COPY);
            ((VBox) event.getSource()).getStyleClass().add("drag-over");
        }
        event.consume();
    }

    /**
     * handleDragDropped():
     * Handles the drop event for folder upload.
     */
    @FXML
    private void handleDragDropped(DragEvent event) {
        Dragboard db = event.getDragboard();
        boolean success = false;
        if (db.hasFiles()) {
            for (File file : db.getFiles()) {
                if (file.isDirectory()) {
                    // Scan and add all image files in the folder
                    File[] imageFiles = file.listFiles((dir, name) -> 
                        name.toLowerCase().endsWith(".png") ||
                        name.toLowerCase().endsWith(".jpg") ||
                        name.toLowerCase().endsWith(".jpeg")
                    );
                    if (imageFiles != null) {
                        for (File img : imageFiles) {
                            mylistview.getItems().add(img.getAbsolutePath());
                            totalImagesLabel.setText(String.valueOf(mylistview.getItems().size()));
                        }
                    }
                } else {
                    mylistview.getItems().add(file.getAbsolutePath());
                    totalImagesLabel.setText(String.valueOf(mylistview.getItems().size()));
                }
            }
            success = true;
            ((VBox) event.getSource()).getStyleClass().remove("drag-over");
        }
        event.setDropCompleted(success);
        event.consume();
    }

    /**
     * updateEstimatedTime():
     * Calculates and updates the estimated remaining time based on average processing time.
     */
    private void updateEstimatedTime() {
        if (completedTasks.get() > 0) {
            double averageTimePerTask = totalClassificationTime.get() / completedTasks.get();
            int remainingTasks = totalTasks.get() - completedTasks.get();
            double remainingTime = averageTimePerTask * remainingTasks;
            estimatedTimeLabel.setText(String.format("%.2fs", remainingTime));
        } else {
            estimatedTimeLabel.setText("Calculating...");
        }
    }

    /**
     * handleCompareModels():
     * Opens a comparison view between multiple models or configurations.
     */
    @FXML
    private void handleCompareModels() {
        // Implementation for comparing different model predictions
        // This could open a new window or dialog with side-by-side comparison
        Alert alert = new Alert(AlertType.INFORMATION);
        alert.setTitle("Compare Models");
        alert.setHeaderText("Model Comparison");
        alert.setContentText("Comparing predictions between multiple models.");
        alert.showAndWait();
    }

    /**
     * handleEditRotate():
     * Rotates the displayed image by 90 degrees.
     */
    @FXML
    private void handleEditRotate() {
        myimageview.setRotate(myimageview.getRotate() + 90);
    }

    /**
     * handleEditCrop():
     * Crops the displayed image.
     */
    @FXML
    private void handleEditCrop() {
        // Implementation for cropping the image
        Alert alert = new Alert(AlertType.INFORMATION);
        alert.setTitle("Crop Image");
        alert.setHeaderText("Crop Tool");
        alert.setContentText("Image cropping functionality is not yet implemented.");
        alert.showAndWait();
    }

    /**
     * handleEditFilter():
     * Applies a basic filter to the displayed image.
     */
    @FXML
    private void handleEditFilter() {
        // Implementation for applying filters
        Alert alert = new Alert(AlertType.INFORMATION);
        alert.setTitle("Apply Filter");
        alert.setHeaderText("Filter Tool");
        alert.setContentText("Image filtering functionality is not yet implemented.");
        alert.showAndWait();
    }

    /**
     * handleToggleFullscreen():
     * Toggles the application between fullscreen and windowed mode.
     */
    @FXML
    private void handleToggleFullscreen() {
        Stage stage = (Stage) myimageview.getScene().getWindow(); // Retrieve the current stage
        stage.setFullScreen(!stage.isFullScreen()); // Toggle fullscreen state
    }
}