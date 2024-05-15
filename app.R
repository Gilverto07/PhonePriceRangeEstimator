library(shiny)
library(data.table)
library(shinythemes)
library(randomForest)
library(knitr)

rmdfiles = c("modelMark.Rmd")
sapply(rmdfiles, knit, quiet = T)

# Load randomForest model
model = readRDS("models2.rds")


# User Interface
ui = fluidPage(theme = shinytheme("slate"),
               navbarPage('Phone Price Range Estimator',
                          tabPanel("Estimator",
                                   sidebarPanel( 
                                     HTML("<h3>Phone Specs</h3>"),
                                     numericInput("battery_power", label = "Battery Power", value = 1021),
                                     numericInput("int_memory", label = "Internal Memory (GB's):", value = 53),
                                     numericInput("mobile_wt", label = "Mobile Weight:", value = 136),
                                     numericInput("px_height", label = "Pixel Resolution Height:", value = 905),
                                     numericInput("px_width", label = "Pixel Resolution Width:", value = 1988),
                                     numericInput("ram", label = "Ram(Mb's):", value = 2631),
                                     actionButton("submitButton", "Submit", class = "btn btn-primary")
                                   ), # end sidebarPanel
                                   
                                   mainPanel(
                                     HTML("<h3>Status/Output</h3>"),
                                     verbatimTextOutput('contents'),
                                     tableOutput('tabledata')
                                   ) # end mainPanel
                                   ), # end tabPanel
                          tabPanel("Model", withMathJax(includeMarkdown("modelMark.md"))
                          ) # end tabPanel
               
                        ) # end navbarPage
)# end fluidPage

# Server
server = function(input, output, session){
  datasetInput = reactive({
    df = data.frame(
      Name = c("battery_power", "int_memory", "mobile_wt","px_height", "px_width", "ram" ),
      Value = as.character(c(input$battery_power, input$int_memory, input$mobile_wt, input$px_height, input$px_width, input$ram)),
      stringsAsFactors = FALSE)
    
    # Save Input into a CSV
    count = 0
    df = rbind(df, count)
    input = transpose(df)
    write.table(input, "input.csv", sep=",", quote = FALSE, row.names = FALSE, col.names = FALSE)
    
    test = read.csv(paste("input", ".csv", sep = ""), header = TRUE)
    
    Output = data.frame(Prediction = predict(model,test), round(predict(model, test, type = "prob"),3))
    print(Output)
  })
  
  # Status/Output text box
  output$contents = renderPrint({
    if(input$submitButton > 0) {
      isolate("Calculation complete.")
    }else {
      return("Server is ready for calculation.")
    }
  }) 
  
  # Prediction results table
  output$tabledata = renderTable({
    if(input$submitButton > 0){
      isolate(datasetInput())
    }
  })
}
shinyApp(ui = ui, server = server)