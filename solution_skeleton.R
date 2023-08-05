
# Import relevant packages ------------------------------------------------

library(tidyverse)
library(plotly)

# Read in the data --------------------------------------------------------

returns_train <- read_csv("data/returns_train.csv")
returns_test <- read_csv("data/returns_test.csv")

# Functions ---------------------------------------------------------------

# We need a function to generate a dataframe/tibble of equal weights. This
# "weights" dataframe must have the same dimensions as the returns dataframe.
# Each weight across each month is calculated as 1/p, where p is the number
# of stocks in the sample.

#' A function to generate equal weights
#'
#' @param data A dataframe, tibble or data.table
#'
#' @return same as input
equalise_weights <- function(data){

  data |>
    pivot_longer(contains("Stock")) |>
    group_by(month_end) |>
    mutate(value = 1/n()) |>
    ungroup() |>
    pivot_wider()

}


#' A function to generate your portflio
#'
#' Function to generate stocks weight allocation for time t+1
#' using historic data. Initial weights generated as 1/p for
#' active stock within a month
#'
#' @param training_data A data.frame, tibble or data.table
#' @param test_data A data.frame, tibble or data.table
#'
#' @return Same as input
generate_portfolio <- function(training_data, test_data){

  message(paste("Portfolio training data ranges from", min(training_data$month_end), 
          "to", max(training_data$month_end)))

  message(paste("Portfolio test data ranges from", min(test_data$month_end),
          "to", max(test_data$month_end)))

# YOUR CODE GOES BELOW THIS LINE ------------------------------------------

  # This is your playground. Delete/modify any of the code here and replace
  #with your methodology. Below we provide a simple, naive estimation to
  # illustrate how we think you should go about structuring your submission
  # and your comments:

  data <- bind_rows(training_data, test_data)

  # We use a static Inverse Volatility Weighting (https://en.wikipedia.org/wiki/Inverse-variance_weighting) 
  # strategy to generate portfolio weights.
  data_long <- 
    data |> 
    pivot_longer(
      contains("Stock"), 
      names_to = "stock", 
      values_to = "value"
    ) # pivot data to long format (columns: month_end, stock, value)

  # To avoid look-ahead bias we calculate the weight for each stock in each
  # month using only returns going up until the date in question. We split
  # this into two steps: 1) We create a function to calculate portfolio weights
  # and then 2) iterate this function over the test data with purrr::map_df() using
  # the test data's month end dates to guide it. See it as a vectorised loop. Read more
  # on purrr here: https://purrr.tidyverse.org/

  calculate_weight <- function(date) {
    data_long |>
      filter(month_end < date) |>
      summarise(vol = sd(value), .by = stock) |> # calculate standard deviation (volatility) for each stock's returns
      mutate(
        inv_vol = 1/vol,              # calculate the inverse of the volatility                          
        tot_inv_vol = sum(inv_vol),   # create a new column = the sum of the volatilities
        weight = inv_vol/tot_inv_vol  # create a weight column
      ) |>
      mutate(month_end = date) |> # add back the date
      select(month_end, name = stock, value = weight) |>
      pivot_wider()
  } 

  df_weights <- map_df(
    .x = test_data$month_end,
    .f = calculate_weight
  )

# YOUR CODE GOES ABOVE THIS LINE ------------------------------------------

  # Your final weights dataframe should have a month_end column, followed by
  # columns of stocks each containing a weight for each date. e.g:
  #
  #   month_end   Stock1 Stock10  Stock11  Stock14 Stock16  Stock18   Stock19
  # 1 2010-01-31  0.0426 -0.0761 -0.150   -0.0313  -0.0530 -0.114   -0.0717
  # 2 2010-02-28 -0.0150 -0.100  -0.0233   0.00248  0.0771  0.0108   0.0224
  # 3 2010-03-31  0.112   0.0979  0.122    0.0644   0.107   0.0179   0.000892
  # 4 2010-04-30 -0.0405 -0.0343 -0.0328  -0.0302   0.0224  0.0855  -0.00716
  # 5 2010-05-31 -0.0694 -0.0732  0.00435  0.0108  -0.0255 -0.00189 -0.0125
  # 6 2010-06-30  0.0588  0.0310 -0.0669   0.0633  -0.0286  0.0203  -0.0644

  # We will use only the weights from the test set's earliest date and tack on
  # equal weights from the training set for charting purposes
  data_out <-
    training_data |>
    equalise_weights() |> 
    bind_rows(df_weights) |>
    arrange(month_end)

  # 10% limit check
  if(any(data_out[-1] > 0.101)) stop("One of your weights exceeds the 0.1 limit.")

  return(data_out)

}


#' Plot the total return
#' 
#' Uses the output from generate_portfolio() 
#'
#' @param df_returns a dataframe of returns
#' @param df_portfolio_weights a dataframe of portfolio weights
#' @param return_data 
#'
#' @return
#' plotly html widget
plot_total_return <- function(df_returns, df_portfolio_weights, return_data = FALSE) {

    returns_long <- pivot_longer(df_returns, contains("Stock"), values_to = "return")

    # generate equal weighted benchmark index
    benchmark_return <-
      equalise_weights(df_returns) |>
      pivot_longer(contains("Stock"), values_to = "weight") |>
      left_join(returns_long, by = c("month_end", "name")) |> 
      mutate(indexed = weight*return) |> 
      summarise(benchmark_return = sum(indexed), .by = c("month_end")) |> 
      mutate(benchmark_return = cumprod(1 + benchmark_return)*100) |> 
      pivot_longer(benchmark_return)

    # process the portfolio weight returns
    portfolio_return <-
      df_portfolio_weights |>
      pivot_longer(contains("Stock"), values_to = "weight") |> 
      left_join(returns_long, by = c("month_end", "name")) |> 
      mutate(indexed = weight*return) |>
      summarise(portfolio_return = sum(indexed), .by = c("month_end")) |> 
      mutate(portfolio_return = cumprod(1 + portfolio_return)*100) |> 
      pivot_longer(2)

    chart_data <- 
      bind_rows(benchmark_return, portfolio_return) |>
      mutate(name = stringr::str_to_title(gsub(pattern = "_", replacement = " ", x = name)))

    if (return_data) return(chart_data)

    plot <- 
      chart_data |>
      group_by(name) |>
      ggplot(aes(x = month_end, y = value, group = name, colour = name)) +
      geom_line() +
      labs(y = "Total Return", x = "Month End")

    ggplotly(plot)

  }

# Run the solution --------------------------------------------------------

returns <- bind_rows(returns_train, returns_test)
portfolio_weights <- generate_portfolio(returns_train, returns_test)
plot_total_return(returns, portfolio_weights, return_data = FALSE)
