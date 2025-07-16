package com.example.imagec2;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import javafx.scene.input.KeyCombination;

import java.io.IOException;

public class Main extends Application {

    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("MainView.fxml"));

        Scene scene = new Scene(fxmlLoader.load());
        scene.getStylesheets().add(getClass().getResource("MainView.css").toExternalForm());
        Image icon = new Image(Main.class.getResourceAsStream("/com/example/imagec2/monstre.png"));
        stage.getIcons().add(icon);
        stage.setTitle("Image Classifier Pro");
        stage.setHeight(1000);
        stage.setFullScreen(true); // Enable full-screen mode
        stage.setFullScreenExitHint(""); // Remove exit hint
        stage.setFullScreenExitKeyCombination(KeyCombination.NO_MATCH); // Disable exit via key
        stage.setResizable(true);
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}
