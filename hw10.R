# ZOO 800 - Homework Week 10
# Nurul islam

# ---
# Objective 1: Real Data Analysis (Deer Serology)
# ---

# A. Find and Load Data

## =========================================================
## 1) LOAD DATA  (CSV in the same folder as this script)
## =========================================================
fname <- "deer_data_14oct10-45.csv"
if (!file.exists(fname)) {
  stop("Data file not found: ", fname, "\nMake sure the CSV is in the repo root next to this script.")
}
deer_data <- read.csv(fname, stringsAsFactors = FALSE)

cat("Columns in file:\n"); print(names(deer_data))

# 2. Process the loaded data for regression
# We select the two continuous variables we need:
# X = th.age (numeric age)
# Y = log10 (log10 transformed BPV titer value)
my_data <- data.frame(
  X_age = deer_data$th.age,
  Y_log10_titer = deer_data$log10
)

# 3. Remove any rows that have missing data (NA)
my_data <- na.omit(my_data)

# 4. Check the results
print(paste("Loaded and filtered data. We have", nrow(my_data), "paired observations."))
print("--- Head of final data for Objective 1: ---")
print(head(my_data))

# Our variables are:
# Y (Response): Y_log10_titer (Log10 transformed BPV titer value)
# X (Predictor): X_age (Estimated deer age in years)
# We hypothesize a causal association that deer age (X) influences
# the accumulated antibody response (Y).
# We have 112 paired observations, which is > 30.

# ---
# B. Fit Linear Regression
# ---
# We fit a linear model of log10_titer predicted by age.
model_obj1 <- lm(Y_log10_titer ~ X_age, data = my_data)
print("--- Objective 1B: Model Summary ---")
summary(model_obj1)

# ---
# C. Evaluate Model Assumptions
# ---
print("--- Objective 1C: Generating Diagnostic Plots ---")

# Create a 2x2 plot layout to see all 4 plots at once
par(mfrow = c(2, 2))
plot(model_obj1)
# Reset plot layout to 1x1
par(mfrow = c(1, 1))

# --- Analysis of Assumptions ---

# Assumption 1: Linearity
# How we check: Using the 'Residuals vs Fitted' plot (top-left).
#
# Analysis: The assumption is moderately violated. The red line shows
# a slight 'U' shape, suggesting a non-linear pattern might be
# present. The vertical stacking of points is an artifact of
# our Y-variable (log10_titer) being discrete (1.0, 1.3, etc.)
# rather than truly continuous.

# Assumption 2: Normality of Residuals
# How we check: Using the 'Normal Q-Q' plot (top-right).
#
# Analysis: The assumption of normality is severely violated.
# The points peel away from the dashed line dramatically at the top
# right. This S-shape, and especially the heavy top tail, shows
# that our residuals are not normally distributed; they are
# heavily right-skewed.

# Assumption 3: Homoscedasticity (Constant Variance)
# How we check: Using the 'Scale-Location' plot (bottom-left).
#
# Analysis: This assumption is also violated. The red line
# trends clearly downward, and the spread of points is wider
# (higher variance) on the left (lower fitted values) and
# tighter (lower variance) on the right. This indicates
# heteroscedasticity (non-constant variance).

# ---
# D. Generate Predictions
# ---

# Find the median and 95th percentile of X (Age)
median_x <- median(my_data$X_age, na.rm = TRUE)
p95_x <- quantile(my_data$X_age, probs = 0.95, na.rm = TRUE)

print(paste("Median X (Age):", median_x))
print(paste("95th Percentile X (Age):", round(p95_x, 2)))

# Create a data frame with these new X values
new_x_values <- data.frame(X_age = c(median_x, p95_x))

# Generate predictions and 95% prediction intervals
pred_intervals <- predict(model_obj1, newdata = new_x_values,
                          interval = "prediction", level = 0.95)

print("--- Objective 1D: Predictions and Intervals ---")
print(cbind(new_x_values, pred_intervals))

# Comparison of prediction intervals:
#
# Analysis: The prediction interval for the 95th percentile age (12)
# is [0.699, 1.503], with a width of 0.804.
# The interval for the median age (3) is [0.682, 1.470],
# with a width of 0.788.
#
# The interval for the more extreme age (12) is slightly wider.
# This makes sense: our model is more uncertain when making
# predictions for X-values that are far from the center of the data.


# ---
# Objective 2: Simulation
# ---
# Evaluate robustness of parameter estimates and uncertainty
# to non-normal errors.

print("--- Objective 2: Starting Simulation (may take a moment) ---")

# Set up simulation parameters
n_sims <- 1000  # Number of repeated simulations
n_pairs <- 100   # 100 X,Y pairs
true_intercept <- 5
true_slope <- 2

# Create vectors to store results
est_intercepts <- numeric(n_sims)
est_slopes <- numeric(n_sims)
prediction_coverage <- numeric(n_sims) # To store the fraction of Ys in the PI

# Set seed for reproducibility
set.seed(42)

# Run the simulation loop
for (i in 1:n_sims) {
  # 1. Generate Data
  # Generate 100 X values (no error in X)
  x_sim <- runif(n_pairs, 0, 10)
  
  # Generate non-normally distributed error from a t-distribution.
  # This has a mean of 0 but very "fat tails" (not normal).
  errors <- rt(n_pairs, df = 3)
  
  # Generate Y values
  y_sim <- true_intercept + true_slope * x_sim + errors
  sim_data <- data.frame(X = x_sim, Y = y_sim)
  
  # 2. Fit linear regression
  model_obj2 <- lm(Y ~ X, data = sim_data)
  
  # 3. Store estimated slope and intercept
  est_intercepts[i] <- coef(model_obj2)[1]
  est_slopes[i] <- coef(model_obj2)[2]
  
  # 4. Generate 95% PIs for each X value
  pis <- predict(model_obj2, interval = "prediction", level = 0.95)
  
  # 5. Check what fraction of Y values fall within the PI
  is_inside <- (sim_data$Y >= pis[, "lwr"]) & (sim_data$Y <= pis[, "upr"])
  prediction_coverage[i] <- sum(is_inside) / n_pairs
}

print("--- Simulation Complete ---")

# ---
# Analysis of Objective 2
# ---

# Question: How well do the estimated slope and intercept match the true values?

print("--- Objective 2: Parameter Estimate Analysis ---")
print(paste("True Intercept:", true_intercept))
print(paste("Mean Estimated Intercept:", round(mean(est_intercepts), 3)))
print(paste("True Slope:", true_slope))
print(paste("Mean Estimated Slope:", round(mean(est_slopes), 3)))

# Plot histograms of estimated parameters
par(mfrow = c(1, 2)) # One row, two columns

hist(est_intercepts, main = "Distribution of Est. Intercepts",
     xlab = "Intercept Value", breaks = 30)
abline(v = true_intercept, col = "red", lwd = 2)

hist(est_slopes, main = "Distribution of Est. Slopes",
     xlab = "Slope Value", breaks = 30)
abline(v = true_slope, col = "red", lwd = 2)

par(mfrow = c(1, 1)) # Reset plot layout

# Analysis: The parameter estimates are highly robust.
# The histograms are centered on the true values (red lines), and
# the printed means of the estimated intercepts and slopes are
# extremely close to the true values (5 and 2). This shows that
# lm() parameter estimates are unbiased even with fat-tailed,
# non-normal errors.


# Question: What fraction of your data (Y values) falls within the 95% PI?

mean_coverage <- mean(prediction_coverage)

print("--- Objective 2: Prediction Interval Coverage ---")
print(paste("Average 95% PI Coverage:", round(mean_coverage, 3)))

# Plot the distribution of coverage fractions
hist(prediction_coverage, main = "Distribution of PI Coverage Fractions",
     xlab = "Proportion of Ys in 95% PI", breaks = 30)
abline(v = 0.95, col = "red", lwd = 2)
abline(v = mean_coverage, col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Target (0.95)", "Actual Mean"),
       col = c("red", "blue"), lty = c(1, 2), lwd = 2)

# Analysis: The mean coverage (printed above, ~0.90 or 90%)
# is well below the 95% target (red line). This shows that
# on average, the 95% prediction intervals only captured
# about 90% of the true data.


# Question: What does this imply for how your estimated uncertainty compares
# to the true uncertainty?
#
# Analysis: This simulation implies that uncertainty estimates
# are *not* robust to this violation of normality.
# Because the actual coverage (~90%) is less than the
# nominal coverage (95%), it means our *estimated* 95%
# prediction intervals are too narrow (i.e., too optimistic)
# and fail to capture the *true* uncertainty of the data
# generating process, which has fatter tails than the model
# assumes.