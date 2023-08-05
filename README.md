![img](header.png)

# Welcome!

Thank you for deciding to honour us with your participation in the the Prescient Coding Challenge. We trust you've brought your thinking caps and are ready to tackle an interesting problem. Most of all, we hope this stimulates you mentally and that you have a lot of fun. We think the grand prize is worth it.

## The Mission

You are given a set of stocks and their monthly returns (return per stock per month) spanning from January 2010 up to last month. We challenge you to construct a portfolio that yields a higher [total return](https://www.investopedia.com/terms/t/totalreturn.asp) than an equally weighted index (the [benchmark index](https://www.investopedia.com/terms/b/benchmark.asp)). In other words, we want you to programmatically come up with a weighted portfolio index of stock returns each month that does better than if all stock returns in this index were just combined in equal weights through time. We will measure the outperformance on the test data only. 

## The Detail

You are given a training data set of dimensions $n_{train} \times (p + 1)$ and a test data set of dimensions and $n_{test} \times (p + 1)$ where $p$ is the number of stocks and $n$ is the length of the returns series for each stock. The monthly returns are in fractional format. 

The total return, $TR()$, on an equally weighted index for a particular month, $t$, is calculated as

$$TR(t) = 100 \times \Biggr[ \prod_{i=1}^{t\leq(n_{train} + n_{test})}  \biggr[ 1 + \Big( \frac{1}{p} \sum_{k=1}^{p} r_{ik} \Big) \biggr] \Biggr]$$

The challenge is to calculate a set of portfolio weights $w_{tk}$ for each month $t \in \{1,2,3..., n_{train} + n_{test}\}$ and stock $k \in \{1,2,3,...p\}$ such that

$$TR^{w}(t) = 100 \times \Biggr[ \prod_{i=1}^{t\leq(n_{train} + n_{test})}  \biggr[ 1 + \Big( \sum_{k=1}^{p} w_{ik}r_{ik} \Big) \biggr] \Biggr]$$

and

$$TR^{w}(n_{train} + n_{test}) \gt TR(n_{train} + n_{test})$$

where $\boldsymbol{w}$ is the matrix of weights of $w_{tk}$ and 

$$0 \leq w_{tk} \le 0.1 \: \forall \:t,k$$
$$ \sum_{k=1}^{p} w_{tk} = 1 \: \forall \:t$$

That is, the latest value in the portfolio's total return series is strictly greater than the corresponding value in the benchmark index's total return series.

## The Rules

1. As stated in the constraints above, no stock may have a weight higher than 10% (it can have a zero weight) at any point in time. Also, all weights must sum to 1 for any given month.
2. You are welcome and encouraged to use any external resources at your disposal (chatGPT, Google, stackoverflow, etc.).
3. You can use any methodology, machine learning, optimisation or algorithm of your choice. Make sure you have comments that explain what you are doing and *why*. The better we understand what you are attempting, the higher your chances of success.
4. You can use any IDE of your choice (VScode, spyder, RStudio, etc). It is the solution template that will matter at the end of the day.
5. We will evaluate your submission based on the following criteria:
   - Your methodology's performance against the equally weighted benchmark index.
   - How well you are able to articulate and explain your own answer. (Hint: Don't just throw all sorts of technical jargon onto a page. You don't have a lot of time. Be concise, show us how much of your answer you understand well.)
   - Generative AI is useful, but we will bias our evaluation towards more original answers. If you do use AI, use it to help you work smart. Don't try to have it do all the work for you.
6. We will also be keeping an eye out for [look-ahead bias](https://www.investopedia.com/terms/l/lookaheadbias.asp), hence the separation between test and training sets.

## The Data

You will find the data split into two `.csv` files, `returns_train.csv` and `returns_test.csv`:

- The data sets are not necessarily ordered by month end.
- The training set is the first part of the data, and the test set the second.
- There is a return for each stock at each month end.

## The Input

You are provided with a solution template to submit your answer in. If you are using R, work in `solution_skeleton.R`. For python, use `solution_skeleton.py`. The solution template contains three functions:

1. `equalise_weights()`: Generates a dataframe of equal weights for the index. 
2. `generate_portfolio()`: The skeleton or sample code to generate the portfolio weights.
3. `plot_total_return()`: Plot the total return series of both the index and portfolio.

The recomendation is to modify the weight generation section in the `generate_portfolio()` function (see below)

for python

```python
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>

# This is your playground. Delete/modify any of the code here and replace with 
# your methodology. Below we provide a simple, naive estimation to illustrate 
# how we think you should go about structuring your submission and your comments:

# We use a static Inverse Volatility Weighting (https://en.wikipedia.org/wiki/Inverse-variance_weighting) 
# strategy to generate portfolio weights.
# Use the latest available data at that point in time

for i in range(len(df_test)):

    # latest data at this point
    df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]
            
    # vol calc
    df_w = pd.DataFrame()
    df_w['vol'] = df_latest.std(numeric_only=True)          # calculate stock volatility
    df_w['inv_vol'] = 1/df_w['vol']                         # calculate the inverse volatility
    df_w['tot_inv_vol'] = df_w['inv_vol'].sum()             # calculate the total inverse volatility
    df_w['weight'] = df_w['inv_vol']/df_w['tot_inv_vol']    # calculate weight based on inverse volatility
    df_w.reset_index(inplace=True, names='name')

    # add to all weights
    df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight'].to_list()], columns=df_latest.columns)
    df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)

# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
```
for R

```r
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
```

## The Output

Once you have the function generating the weights you can use the `plot_total_return` to compare the index and your portfolio output. Please keep the output consistent so that you don't have to recode the plotting function.

# Github Logistics

## Getting the project on your computer

Assuming you have a Github account and git installed on your computer, follow these steps:

1. Log in to Github
2. Go to https://github.com/PIM-Data-Science/prescient-coding-challenge-2023
3. Fork the repository
4. On your version of the project, click the `Code` button and copy the link under the _HTTPS_ header.
5. On your computer, go to the location you would like to store this project
6. In this location, right click and open Git Bash if on Windows, or Terminal if on Mac. 
7. In terminal, type and run `git clone your-copied-link`
8. Open the project in your preferred IDE and start coding!

## How to submit your answer

1. Once everything has been saved, open Git Bash (or terminal) again in that same location
2. Run `git status`. You should see your file listed as modified.
3. Run `git commit -am "Insert your team members' names here."`
4. Run `git push`
5. Go to your github profile
6. Open the project online, you should see your changes
7. Open _Pull Requests_ and then click on `New Pull Request`
8. Add your team's names to the pull request subscription, and finish creating the pull request.  

# Download Links

1. [Git](https://git-scm.com/downloads)
2. [Python ](https://www.python.org/downloads/)
3. [VS Code](https://code.visualstudio.com/download)
4. [R Base](https://cran.r-project.org/)
5. [R Studio](https://posit.co/downloads/)
