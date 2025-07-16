module com.example.imageclass {
    requires javafx.controls;
    requires javafx.fxml;
    requires ai.djl.api;
    requires ai.djl.basicdataset;
    requires org.controlsfx.controls;
    requires javafx.base;
    requires javafx.graphics;
    requires javafx.web;
    requires ai.djl.pytorch_engine;

    opens com.example.imagec2 to javafx.fxml;

    exports com.example.imagec2;
}