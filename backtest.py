import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
INITIAL_CAPITAL = 50000
TOP_N_STOCKS = 22
REBALANCE_THRESHOLD = 0.01  # 1% change in shares
PROGRESS_INTERVAL = 2000

def parse_ohlc_data(all_data, close_df):
    """Parse OHLC data from all_stock_data.csv in various formats"""
    cols = all_data.columns.tolist()
    
    # Check if multi-index columns (stock, OHLC)
    if isinstance(all_data.columns, pd.MultiIndex):
        print("Detected multi-index column format (stock, OHLC)")
        try:
            open_df = all_data.xs('Open', level=1, axis=1)
            high_df = all_data.xs('High', level=1, axis=1)
            low_df = all_data.xs('Low', level=1, axis=1)
            return open_df, high_df, low_df
        except:
            pass
    
    # Check if long format (date, stock, Open, High, Low, Close)
    if any(c.lower() in ['open', 'high', 'low', 'close', 'o', 'h', 'l', 'c'] for c in cols):
        stock_col = None
        for name in ['asset', 'stock', 'symbol', 'ticker', 'ticker_symbol']:
            if name in cols:
                stock_col = name
                break
        
        date_col = None
        for name in ['datetime', 'date', 'Date', 'DateTime']:
            if name in cols:
                date_col = name
                break
        
        if stock_col and date_col:
            print("Detected long format, pivoting...")
            open_col = next((c for c in cols if c.lower() in ['open', 'o']), None)
            high_col = next((c for c in cols if c.lower() in ['high', 'h']), None)
            low_col = next((c for c in cols if c.lower() in ['low', 'l']), None)
            
            if open_col and high_col and low_col:
                # Parse datetime if it's a string
                if all_data[date_col].dtype == 'object':
                    all_data[date_col] = pd.to_datetime(all_data[date_col])
                
                open_df = all_data.pivot_table(index=date_col, columns=stock_col, values=open_col)
                high_df = all_data.pivot_table(index=date_col, columns=stock_col, values=high_col)
                low_df = all_data.pivot_table(index=date_col, columns=stock_col, values=low_col)
                
                # Set index name to None for consistency
                open_df.index.name = None
                high_df.index.name = None
                low_df.index.name = None
                
                return open_df, high_df, low_df
    
    # Check if wide format with pattern: STOCK_Open, STOCK_High, etc.
    stocks = close_df.columns
    open_dict = {}
    high_dict = {}
    low_dict = {}
    
    for stock in stocks:
        # Try various naming patterns
        for suffix in ['_Open', '_O', '_open', '_o']:
            col_name = stock + suffix
            if col_name in cols:
                open_dict[stock] = all_data[col_name]
                break
        
        for suffix in ['_High', '_H', '_high', '_h']:
            col_name = stock + suffix
            if col_name in cols:
                high_dict[stock] = all_data[col_name]
                break
        
        for suffix in ['_Low', '_L', '_low', '_l']:
            col_name = stock + suffix
            if col_name in cols:
                low_dict[stock] = all_data[col_name]
                break
    
    if open_dict and high_dict and low_dict:
        print("Detected wide format with stock_OHLC pattern")
        open_df = pd.DataFrame(open_dict, index=all_data.index)
        high_df = pd.DataFrame(high_dict, index=all_data.index)
        low_df = pd.DataFrame(low_dict, index=all_data.index)
        return open_df, high_df, low_df
    
    return None, None, None

def load_data():
    """Load close price data and OHLC data if available"""
    print("Loading data...")
    
    # Load close prices
    close_df = pd.read_csv('close_price_data.csv', index_col=0, parse_dates=True)
    print(f"Loaded close prices: {close_df.shape[0]} days, {close_df.shape[1]} stocks")
    
    # Try to load OHLC from all_stock_data.csv
    try:
        print("Loading OHLC data from all_stock_data.csv...")
        # Load without index_col first to check structure
        all_data = pd.read_csv('all_stock_data.csv')
        print(f"Loaded all_stock_data: {all_data.shape[0]} rows, {all_data.shape[1]} columns")
        print(f"Sample columns: {all_data.columns.tolist()[:10]}")
        
        open_df, high_df, low_df = parse_ohlc_data(all_data, close_df)
        
        if open_df is not None and high_df is not None and low_df is not None:
            # Align with close_df dates and columns
            common_dates = close_df.index.intersection(open_df.index)
            common_stocks = close_df.columns.intersection(open_df.columns)
            
            if len(common_dates) > 0 and len(common_stocks) > 0:
                open_df = open_df.loc[common_dates, common_stocks]
                high_df = high_df.loc[common_dates, common_stocks]
                low_df = low_df.loc[common_dates, common_stocks]
                
                print(f"Successfully extracted OHLC data: {len(common_dates)} days, {len(common_stocks)} stocks")
                return close_df, (open_df, high_df, low_df)
            else:
                print("Warning: No common dates/stocks between close_df and OHLC data")
        else:
            print("Could not parse OHLC structure from all_stock_data.csv")
            
    except Exception as e:
        print(f"Error loading all_stock_data.csv: {e}")
        import traceback
        traceback.print_exc()
    
    print("Will approximate VWAP from close prices")
    return close_df, None

def calculate_vwap(close_df, ohlc_data=None):
    """Calculate VWAP = average(OHLC) for each stock each day"""
    if ohlc_data is not None:
        open_df, high_df, low_df = ohlc_data
        # VWAP = (Open + High + Low + Close) / 4
        vwap = (open_df + high_df + low_df + close_df) / 4
        # Align indices in case of mismatches
        vwap = vwap.reindex(close_df.index).reindex(columns=close_df.columns)
        vwap = vwap.fillna(close_df)  # Fallback to close if missing
    else:
        # Approximate VWAP as close price (since we only have close data)
        # In real scenario, VWAP = (O + H + L + C) / 4
        vwap = close_df.copy()
    
    return vwap

def calculate_ranks(vwap, close):
    """Calculate rank(vwap-close) and rank(vwap+close) across all stocks for each day"""
    # Calculate vwap - close and vwap + close
    vwap_minus_close = vwap - close
    vwap_plus_close = vwap + close
    
    # Rank across stocks for each day (rank=1 for smallest value)
    # Using method='min' so rank 1 = smallest value
    rank_vwap_minus_close = vwap_minus_close.rank(axis=1, method='min', na_option='keep')
    rank_vwap_plus_close = vwap_plus_close.rank(axis=1, method='min', na_option='keep')
    
    return rank_vwap_minus_close, rank_vwap_plus_close

def calculate_weights(rank_vwap_minus_close, rank_vwap_plus_close):
    """Calculate weights: rank(vwap-close)/rank(vwap+close) / SUM(...)"""
    # Calculate ratio for each stock
    ratio = rank_vwap_minus_close / rank_vwap_plus_close
    
    # Sum across all stocks for each day
    sum_ratio = ratio.sum(axis=1)
    
    # Normalize: divide by sum
    weights = ratio.div(sum_ratio, axis=0)
    
    return weights

def select_top_stocks(weights, top_n=TOP_N_STOCKS):
    """Select top N stocks by weight and normalize weights"""
    # Get top N stocks for each day
    top_weights = weights.copy()
    
    # For each day, keep only top N stocks, set others to 0
    for date in top_weights.index:
        day_weights = top_weights.loc[date]
        # Get top N stocks
        top_n_indices = day_weights.nlargest(top_n).index
        # Set all others to 0
        top_weights.loc[date, ~top_weights.columns.isin(top_n_indices)] = 0
    
    # Normalize weights (only for selected stocks)
    sum_weights = top_weights.sum(axis=1)
    top_weights = top_weights.div(sum_weights, axis=0)
    
    return top_weights

def run_backtest(close_df, weights_df):
    """Run the backtest with rebalancing logic"""
    dates = close_df.index
    stocks = close_df.columns
    
    # Initialize portfolio
    capital = INITIAL_CAPITAL
    shares = pd.Series(0.0, index=stocks)  # Shares held for each stock
    portfolio_value = []
    shares_tracking = []  # Track target shares and actual shares
    
    # Track commission and margin usage
    total_commission = 0.0
    margin_days = 0  # Count days when capital is negative (using margin)
    
    # Track previous target shares for rebalancing
    prev_target_shares = pd.Series(0.0, index=stocks)
    
    print(f"\nStarting backtest from {dates[0]} to {dates[-1]}")
    print(f"Total days: {len(dates)}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}\n")
    
    for i, date in enumerate(dates):
        # Get current prices
        prices = close_df.loc[date]
        
        # Get target weights for this day
        if date in weights_df.index:
            target_weights = weights_df.loc[date]
        else:
            # If no weights for this date, skip rebalancing
            target_weights = pd.Series(0.0, index=stocks)
        
        # Calculate current portfolio value (cash + holdings)
        holdings_value = (shares * prices).sum()
        total_value = capital + holdings_value
        if total_value <= 0:
            total_value = INITIAL_CAPITAL  # Fallback
        
        # Calculate target dollar amounts
        target_dollars = target_weights * total_value
        
        # Calculate target shares
        target_shares = target_dollars / prices
        target_shares = target_shares.fillna(0)
        target_shares = target_shares.replace([np.inf, -np.inf], 0)
        
        # Check which stocks need rebalancing (>1% change in shares)
        # Calculate percentage change: |new_shares - old_shares| / old_shares
        for stock in stocks:
            current_shares = shares[stock]
            target_share_count = target_shares[stock]
            
            # Skip if both are zero or very small
            if abs(current_shares) < 1e-10 and abs(target_share_count) < 1e-10:
                continue
            
            # Calculate percentage change
            if abs(current_shares) > 1e-10:
                pct_change = abs(target_share_count - current_shares) / abs(current_shares)
            else:
                # If current shares is 0 but target is not, that's a 100% change (definitely rebalance)
                pct_change = 1.0 if abs(target_share_count) > 1e-10 else 0.0
            
            # Rebalance if change > 1%
            if pct_change > REBALANCE_THRESHOLD:
                # Calculate needed shares change
                shares_diff = target_share_count - current_shares
                cost = shares_diff * prices[stock]
                
                # Calculate commission: max(0.39, 3bp of traded value)
                traded_value = abs(shares_diff * prices[stock])
                commission = max(0.39, traded_value * 0.0003)  # 3bp = 0.03% = 0.0003
                total_cost = cost + commission
                total_commission += commission  # Track total commission paid
                
                # Execute trade (treat as margin, so allow negative cash - no normalization)
                shares[stock] = target_share_count
                capital -= total_cost
        
        # Track margin usage (when capital is negative)
        if capital < 0:
            margin_days += 1
        
        # Update portfolio value
        portfolio_value.append({
            'date': date,
            'value': (shares * prices).sum() + capital,
            'cash': capital,
            'holdings_value': (shares * prices).sum()
        })
        
        # Track target shares and actual shares with rebalance logic
        shares_tracking.append({
            'date': date,
            'target_shares': target_shares.copy(),
            'actual_shares': shares.copy()
        })
        
        # Print progress every 2000 days
        if (i + 1) % PROGRESS_INTERVAL == 0:
            current_portfolio_value = (shares * prices).sum() + capital
            print(f"Day {i+1}/{len(dates)} ({date.strftime('%Y-%m-%d')}): "
                  f"Portfolio Value: ${current_portfolio_value:,.2f}, "
                  f"Cash: ${capital:,.2f}, "
                  f"Holdings: ${(shares * prices).sum():,.2f}")
    
    portfolio_df = pd.DataFrame(portfolio_value)
    shares_tracking_df = pd.DataFrame(shares_tracking)
    
    # Calculate margin % time
    total_days = len(dates)
    margin_pct_time = (margin_days / total_days * 100) if total_days > 0 else 0.0
    
    return portfolio_df, shares, shares_tracking_df, total_commission, margin_pct_time

def calculate_cagr(values_df, start_date, end_date):
    """Calculate Compound Annual Growth Rate"""
    if len(values_df) < 2:
        return 0.0
    
    initial_value = values_df.iloc[0]
    final_value = values_df.iloc[-1]
    
    if initial_value <= 0:
        return 0.0
    
    # Calculate years
    if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
        years = (end_date - start_date).days / 365.25
    else:
        years = len(values_df) / 252  # Approximate trading days per year
    
    if years <= 0:
        return 0.0
    
    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr * 100  # Return as percentage

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.0):
    """Calculate annualized Sharpe Ratio"""
    if len(returns_series) < 2:
        return 0.0
    
    # Calculate daily returns if not already
    if isinstance(returns_series, pd.Series):
        returns = returns_series.pct_change().dropna()
    else:
        returns = pd.Series(returns_series).pct_change().dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    # Annualize
    mean_return = returns.mean() * 252  # Annualized mean
    std_return = returns.std() * np.sqrt(252)  # Annualized std
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

def calculate_mdd(values_series):
    """Calculate Maximum Drawdown"""
    if len(values_series) < 2:
        return 0.0
    
    if isinstance(values_series, pd.Series):
        values = values_series.values
    else:
        values = np.array(values_series)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdown
    drawdown = (values - running_max) / running_max
    
    # Maximum drawdown (most negative)
    mdd = drawdown.min() * 100  # Return as percentage
    return abs(mdd)  # Return as positive percentage

def calculate_sortino_ratio(returns_series, risk_free_rate=0.0):
    """Calculate annualized Sortino Ratio"""
    if len(returns_series) < 2:
        return 0.0
    
    # Calculate daily returns if not already
    if isinstance(returns_series, pd.Series):
        returns = returns_series.pct_change().dropna()
    else:
        returns = pd.Series(returns_series).pct_change().dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Annualize mean return
    mean_return = returns.mean() * 252
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        # If no negative returns, use standard deviation
        downside_std = returns.std() * np.sqrt(252)
    else:
        downside_std = downside_returns.std() * np.sqrt(252)
    
    if downside_std == 0:
        return 0.0
    
    sortino = (mean_return - risk_free_rate) / downside_std
    return sortino

def get_sp500_bh_return(start_date, end_date):
    """Get S&P500 buy and hold return from yfinance"""
    print("\nFetching S&P500 data from yfinance...")
    try:
        sp500 = yf.Ticker("^GSPC")
        # Convert to datetime if needed
        if isinstance(start_date, pd.Timestamp):
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = str(start_date)
        
        if isinstance(end_date, pd.Timestamp):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = str(end_date)
        
        hist = sp500.history(start=start_str, end=end_str)
        
        if len(hist) == 0:
            print("Warning: Could not fetch S&P500 data")
            return None
        
        return {
            'data': hist
        }
    except Exception as e:
        print(f"Warning: Error fetching S&P500 data: {e}")
        return None

def main():
    # Load data
    close_df, all_data = load_data()
    
    # Calculate VWAP
    vwap = calculate_vwap(close_df, all_data)
    
    # Calculate ranks
    rank_vwap_minus_close, rank_vwap_plus_close = calculate_ranks(vwap, close_df)
    
    # Calculate weights
    weights = calculate_weights(rank_vwap_minus_close, rank_vwap_plus_close)
    
    # Select top 22 stocks and normalize
    top_weights = select_top_stocks(weights, top_n=TOP_N_STOCKS)
    
    # Run backtest
    portfolio_df, final_shares, shares_tracking_df, total_commission, margin_pct_time = run_backtest(close_df, top_weights)
    
    # Calculate strategy returns
    initial_value = INITIAL_CAPITAL
    final_value = portfolio_df['value'].iloc[-1]
    strategy_return = (final_value / initial_value - 1) * 100
    
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Capital: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {strategy_return:.2f}%")
    print(f"Total Return (absolute): ${final_value - initial_value:,.2f}")
    print(f"Total Commission Paid: ${total_commission:,.2f}")
    print(f"Margin % Time: {margin_pct_time:.2f}%")
    
    # Calculate performance metrics for strategy
    start_date = close_df.index[0]
    end_date = close_df.index[-1]
    portfolio_values = portfolio_df['value']
    strategy_cagr = calculate_cagr(portfolio_values, start_date, end_date)
    strategy_sharpe = calculate_sharpe_ratio(portfolio_values)
    strategy_mdd = calculate_mdd(portfolio_values)
    strategy_sortino = calculate_sortino_ratio(portfolio_values)
    
    # Fetch S&P500 data and calculate metrics
    sp500_result = get_sp500_bh_return(start_date, end_date)
    sp500_cagr = None
    sp500_sharpe = None
    sp500_mdd = None
    sp500_sortino = None
    
    if sp500_result and 'data' in sp500_result:
        sp500_prices = sp500_result['data']['Close']
        sp500_cagr = calculate_cagr(sp500_prices, start_date, end_date)
        sp500_sharpe = calculate_sharpe_ratio(sp500_prices)
        sp500_mdd = calculate_mdd(sp500_prices)
        sp500_sortino = calculate_sortino_ratio(sp500_prices)
    
    # Display metrics table
    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Strategy':>15} {'S&P500 B&H':>15} {'Difference':>15}")
    print(f"{'-'*80}")
    
    # Format CAGR
    sp500_cagr_str = f"{sp500_cagr:>15.2f}" if sp500_cagr is not None else f"{'N/A':>15}"
    diff_cagr = strategy_cagr - sp500_cagr if sp500_cagr is not None else None
    diff_cagr_str = f"{diff_cagr:>15.2f}" if diff_cagr is not None else f"{'N/A':>15}"
    print(f"{'CAGR (%)':<20} {strategy_cagr:>15.2f} {sp500_cagr_str} {diff_cagr_str}")
    
    # Format Sharpe
    sp500_sharpe_str = f"{sp500_sharpe:>15.2f}" if sp500_sharpe is not None else f"{'N/A':>15}"
    diff_sharpe = strategy_sharpe - sp500_sharpe if sp500_sharpe is not None else None
    diff_sharpe_str = f"{diff_sharpe:>15.2f}" if diff_sharpe is not None else f"{'N/A':>15}"
    print(f"{'Sharpe Ratio':<20} {strategy_sharpe:>15.2f} {sp500_sharpe_str} {diff_sharpe_str}")
    
    # Format MDD
    sp500_mdd_str = f"{sp500_mdd:>15.2f}" if sp500_mdd is not None else f"{'N/A':>15}"
    diff_mdd = strategy_mdd - sp500_mdd if sp500_mdd is not None else None
    diff_mdd_str = f"{diff_mdd:>15.2f}" if diff_mdd is not None else f"{'N/A':>15}"
    print(f"{'Max Drawdown (%)':<20} {strategy_mdd:>15.2f} {sp500_mdd_str} {diff_mdd_str}")
    
    # Format Sortino
    sp500_sortino_str = f"{sp500_sortino:>15.2f}" if sp500_sortino is not None else f"{'N/A':>15}"
    diff_sortino = strategy_sortino - sp500_sortino if sp500_sortino is not None else None
    diff_sortino_str = f"{diff_sortino:>15.2f}" if diff_sortino is not None else f"{'N/A':>15}"
    print(f"{'Sortino Ratio':<20} {strategy_sortino:>15.2f} {sp500_sortino_str} {diff_sortino_str}")
    
    print(f"{'='*80}")
    
    # Create results folder if it doesn't exist
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    
    # Save results
    portfolio_path = os.path.join(results_folder, 'portfolio_results.csv')
    portfolio_df.to_csv(portfolio_path, index=False)
    print(f"\nPortfolio results saved to {portfolio_path}")
    
    # Save target shares and actual shares with rebalance logic
    # Reshape to wide format: date as index, stocks as columns
    target_shares_list = []
    actual_shares_list = []
    dates_list = []
    
    for idx, row in shares_tracking_df.iterrows():
        date = row['date']
        target_shares_series = row['target_shares']
        actual_shares_series = row['actual_shares']
        
        target_dict = {'date': date}
        actual_dict = {'date': date}
        
        for stock in target_shares_series.index:
            target_dict[stock] = target_shares_series[stock]
            actual_dict[stock] = actual_shares_series[stock]
        
        target_shares_list.append(target_dict)
        actual_shares_list.append(actual_dict)
    
    target_shares_df = pd.DataFrame(target_shares_list)
    actual_shares_df = pd.DataFrame(actual_shares_list)
    
    # Set date as index
    target_shares_df.set_index('date', inplace=True)
    actual_shares_df.set_index('date', inplace=True)
    
    # Save to CSV
    target_shares_path = os.path.join(results_folder, 'target_shares.csv')
    actual_shares_path = os.path.join(results_folder, 'actual_shares.csv')
    actual_shares_transposed_path = os.path.join(results_folder, 'actual_shares_transposed.csv')
    
    target_shares_df.to_csv(target_shares_path)
    actual_shares_df.to_csv(actual_shares_path)
    actual_shares_df.T.to_csv(actual_shares_transposed_path)
    
    print(f"Target shares saved to {target_shares_path}")
    print(f"Actual shares (with rebalance logic) saved to {actual_shares_path}")
    print(f"Actual shares transposed saved to {actual_shares_transposed_path}")

if __name__ == "__main__":
    main()

