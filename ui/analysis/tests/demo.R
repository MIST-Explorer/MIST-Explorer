library(ggplot2)

# Generate random data
set.seed(42)  # For reproducibility
data <- data.frame(values = rnorm(1000, mean = 50, sd = 10))

# Create histogram
histogram_plot <- ggplot(data, aes(x = values)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Random Data", x = "Value", y = "Frequency") +
  theme_minimal()

# Display the plot
print(histogram_plot)
