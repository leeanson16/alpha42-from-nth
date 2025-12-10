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
# STOP_LOSS_PCT = 0.05  # 5% stop loss
# STOP_LOSS_INACTIVE_DAYS = 250  # Days to keep stock inactive after stop loss

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

def run_backtest(close_df, weights_df, enable_sp500_short=True):
    """Run the backtest with rebalancing logic
    
    Args:
        close_df: DataFrame with close prices
        weights_df: DataFrame with target weights
        enable_sp500_short: If True, include S&P500 short position
    """
    dates = close_df.index
    stocks = close_df.columns
    
    # Get S&P500 prices for the backtest period (only if short is enabled)
    start_date = dates[0]
    end_date = dates[-1]
    sp500_prices = None
    if enable_sp500_short:
        try:
            sp500 = yf.Ticker("^GSPC")
            if isinstance(start_date, pd.Timestamp):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)
            if isinstance(end_date, pd.Timestamp):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)
            sp500_hist = sp500.history(start=start_str, end=end_str)
            if len(sp500_hist) > 0:
                sp500_prices = sp500_hist['Close']
                # Remove timezone from index to match backtest dates (yfinance returns timezone-aware)
                if sp500_prices.index.tz is not None:
                    # Normalize to date and remove timezone
                    sp500_prices.index = pd.to_datetime(sp500_prices.index.date)
                # Align with backtest dates
                sp500_prices = sp500_prices.reindex(dates, method='ffill')
                print(f"Loaded S&P500 prices for short position: {len(sp500_prices)} days")
        except Exception as e:
            print(f"Warning: Could not load S&P500 prices for short position: {e}")
            os._exit()
    
    # Initialize portfolio
    capital = INITIAL_CAPITAL
    shares = pd.Series(0.0, index=stocks)  # Shares held for each stock
    sp500_short_shares = 0.0  # Short S&P500 shares (positive value, but position is short)
    sp500_short_entry_price = 0.0  # Entry price of short position
    sp500_short_proceeds = 0.0  # Track cash received from short sales (for sizing calculation)
    portfolio_value = []
    shares_tracking = []  # Track target shares and actual shares
    
    # Track commission and margin usage
    total_commission = 0.0
    margin_days = 0  # Count days when capital is negative (using margin)
    
    # Track stop loss: average entry prices and inactive stocks
    average_entry = pd.Series(0.0, index=stocks)  # Average entry price for each stock (weighted average)
    inactive_stocks = {}  # Dict: {stock: reactivation_date} for stocks on stop loss
    stop_loss_triggers = 0  # Count stop loss triggers
    
    # Track previous target shares for rebalancing
    prev_target_shares = pd.Series(0.0, index=stocks)
    
    print(f"\nStarting backtest from {dates[0]} to {dates[-1]}")
    print(f"Total days: {len(dates)}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}\n")
    
    for i, date in enumerate(dates):
        # Get current prices
        prices = close_df.loc[date]
        
        # # Check for stop loss triggers and update inactive stocks
        # stocks_to_reactivate = []
        # for stock in inactive_stocks:
        #     if date >= inactive_stocks[stock]:
        #         stocks_to_reactivate.append(stock)
        # 
        # # Reactivate stocks that have completed their inactive period
        # for stock in stocks_to_reactivate:
        #     del inactive_stocks[stock]
        #     average_entry[stock] = 0.0  # Reset average entry price
        # 
        # # Check for stop loss on currently held stocks (using close price)
        # for stock in stocks:
        #     if stock in inactive_stocks:
        #         continue  # Already inactive
        #     
        #     current_shares = shares[stock]
        #     # Use close price for stop loss check
        #     curr_price = prices[stock]  # prices comes from close_df, so this is close price
        #     avg_entry = average_entry[stock]
        #     
        #     # Check stop loss: curr_price < average_entry * (1 - stop_loss)
        #     if (STOP_LOSS_PCT > 0 and current_shares > 1e-10 and 
        #         avg_entry > 1e-10 and not pd.isna(curr_price) and not pd.isna(avg_entry) and
        #         curr_price < avg_entry * (1 - STOP_LOSS_PCT)):
        #         # Trigger stop loss: sell position and inactivate
        #         stop_loss_triggers += 1
        #         sell_value = current_shares * curr_price
        #         commission = max(0.39, sell_value * 0.0003)
        #         capital += sell_value - commission
        #         total_commission += commission
        #         
        #         # Set shares to 0 and mark as inactive
        #         shares[stock] = 0.0
        #         # Calculate reactivation date as 250 trading days from now
        #         current_date_idx = dates.get_loc(date)
        #         reactivation_idx = min(current_date_idx + STOP_LOSS_INACTIVE_DAYS, len(dates) - 1)
        #         reactivation_date = dates[reactivation_idx]
        #         inactive_stocks[stock] = reactivation_date
        #         average_entry[stock] = 0.0
        
        # Get target weights for this day
        if date in weights_df.index:
            target_weights = weights_df.loc[date].copy()
        else:
            # If no weights for this date, skip rebalancing
            target_weights = pd.Series(0.0, index=stocks)
        
        # # Set weights to 0 for inactive stocks (weight will be held as cash, no normalization)
        # for stock in inactive_stocks:
        #     if stock in target_weights.index:
        #         target_weights[stock] = 0.0
        
        # Calculate base portfolio value (cash + holdings, WITHOUT S&P500 short P&L)
        # This is the long portfolio value that we want to hedge with the short
        holdings_value = (shares * prices).sum()
        base_portfolio_value = capital + holdings_value
        # Allow negative values (margin); only fallback if non-finite
        if not np.isfinite(base_portfolio_value):
            base_portfolio_value = INITIAL_CAPITAL  # Fallback
        # Cap at reasonable maximum to prevent runaway growth
        if base_portfolio_value > INITIAL_CAPITAL * 1000:  # Cap at 1000x initial capital
            base_portfolio_value = INITIAL_CAPITAL * 1000
        
        # Calculate unrealized P&L from existing S&P500 short position
        sp500_short_pnl = 0.0
        if enable_sp500_short and sp500_prices is not None and date in sp500_prices.index and not pd.isna(sp500_prices[date]):
            current_sp500_price = sp500_prices[date]
            if sp500_short_shares > 1e-10 and sp500_short_entry_price > 1e-10:
                # Short P&L: profit when price goes down (entry_price - current_price) * shares
                sp500_short_pnl = sp500_short_shares * (sp500_short_entry_price - current_sp500_price)
        
        # Total value includes S&P500 short P&L (for display and metrics)
        total_value = base_portfolio_value + sp500_short_pnl
        
        # Update S&P500 short position to match BASE portfolio value (not including S&P500 P&L to avoid circular dependency)
        # IMPORTANT: Calculate target size BEFORE any rebalancing to avoid feedback loop
        if enable_sp500_short and sp500_prices is not None and date in sp500_prices.index and not pd.isna(sp500_prices[date]):
            current_sp500_price = sp500_prices[date]
            # Short the base portfolio value (cash + stock holdings, excluding S&P500 P&L)
            # This is calculated BEFORE we add short sale proceeds to avoid feedback loop
            target_sp500_short_value = base_portfolio_value  # Short base portfolio value worth
            # Ensure we don't divide by zero or get inf
            if current_sp500_price > 1e-10 and np.isfinite(target_sp500_short_value) and np.isfinite(current_sp500_price):
                target_sp500_short_shares = target_sp500_short_value / current_sp500_price
            else:
                target_sp500_short_shares = 0.0
            
            # Check if we need to rebalance S&P500 short position (>1% change)
            if sp500_short_shares > 1e-10:
                sp500_pct_change = abs(target_sp500_short_shares - sp500_short_shares) / sp500_short_shares
            else:
                # First time opening short position
                sp500_pct_change = 1.0 if target_sp500_short_shares > 1e-10 else 0.0
            
            if sp500_pct_change > REBALANCE_THRESHOLD or (sp500_short_shares < 1e-10 and target_sp500_short_shares > 1e-10):
                # Realize P&L from old short position before closing
                if sp500_short_shares > 1e-10:
                    # Close old short: need to buy back shares to close position
                    close_value = sp500_short_shares * current_sp500_price
                    commission = max(0.39, close_value * 0.0003)
                    total_commission += commission
                    # When closing: pay to buy back shares
                    # Calculate realized P&L: (entry_price - current_price) * shares - commissions
                    # We had received proceeds when opening (tracked but not added to capital)
                    # Opening commission was already subtracted from capital when we opened
                    # Now we pay to close, and the P&L is the difference
                    realized_pnl = sp500_short_shares * (sp500_short_entry_price - current_sp500_price)
                    # Net capital change: we pay to close, but we had received proceeds when opening
                    # Opening: received (entry_price * shares - commission), paid commission from capital
                    # Closing: pay (current_price * shares + commission)
                    # Net: (entry_price - current_price) * shares - opening_commission - closing_commission
                    capital += realized_pnl - commission  # Add P&L, subtract closing commission
                    sp500_short_proceeds = 0.0  # Reset since position is closed
                
                # Open new short position at current price
                if target_sp500_short_shares > 1e-10:
                    short_value = target_sp500_short_shares * current_sp500_price
                    commission = max(0.39, short_value * 0.0003)
                    total_commission += commission
                    # When shorting, you receive cash from selling borrowed shares (minus commission)
                    # In a market-neutral strategy, these proceeds are margin/collateral, not free cash
                    # We DON'T add them to capital to avoid feedback loop in sizing
                    # But we DO subtract the opening commission from capital
                    capital -= commission  # Pay opening commission
                    short_proceeds = short_value - commission
                    sp500_short_proceeds = short_proceeds  # Track for when we close (but don't add to capital)
                    
                    # Set new short position
                    sp500_short_shares = target_sp500_short_shares
                    sp500_short_entry_price = current_sp500_price  # New entry price is current price
                else:
                    sp500_short_shares = 0.0
                    sp500_short_entry_price = 0.0
                    sp500_short_proceeds = 0.0
            
            # Recalculate base portfolio value after S&P500 rebalancing (capital may have changed from realized P&L)
            base_portfolio_value = capital + holdings_value
            # Allow negative (margin); only fallback if non-finite
            if not np.isfinite(base_portfolio_value):
                base_portfolio_value = INITIAL_CAPITAL  # Fallback
            
            # Recalculate S&P500 short P&L after rebalancing (if position exists)
            if enable_sp500_short and sp500_prices is not None and date in sp500_prices.index and not pd.isna(sp500_prices[date]):
                current_sp500_price = sp500_prices[date]
                if sp500_short_shares > 1e-10 and sp500_short_entry_price > 1e-10:
                    # Recalculate unrealized P&L after rebalancing
                    sp500_short_pnl = sp500_short_shares * (sp500_short_entry_price - current_sp500_price)
                else:
                    sp500_short_pnl = 0.0
            
            # Recalculate total_value after S&P500 rebalancing to include updated P&L and capital changes
            total_value = base_portfolio_value + sp500_short_pnl
        else:
            # No S&P500 data, set P&L to 0
            sp500_short_pnl = 0.0
        
        # Calculate target dollar amounts (using total_value that includes S&P500 short P&L)
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
                prev_shares = current_shares
                shares[stock] = target_share_count
                capital -= total_cost
                
                # Update average entry price (like temp_func.py logic)
                held_current = target_share_count
                curr_price = prices[stock]  # Use close price
                
                if held_current > prev_shares:  # Buying more
                    buy_shares = held_current - prev_shares
                    if prev_shares == 0:
                        # New position
                        average_entry[stock] = curr_price
                    else:
                        # Adding to existing position: weighted average
                        # new_total_cost = prev_shares * average_entry + buy_shares * curr_price
                        prev_avg = average_entry[stock] if average_entry[stock] > 1e-10 else curr_price
                        new_total_cost = prev_shares * prev_avg + buy_shares * curr_price
                        average_entry[stock] = new_total_cost / held_current if held_current > 0 else 0.0
                elif held_current < prev_shares:  # Selling (partial or full)
                    if held_current == 0:
                        # Position fully closed
                        average_entry[stock] = 0.0
                    # Average stays the same for remaining shares (no update needed)
                # else: no change in shares, average unchanged
        
        # Recalculate holdings value after stock rebalancing
        holdings_value_after_rebalancing = (shares * prices).sum()
        
        # Recalculate S&P500 short P&L at end of day (after all rebalancing)
        # IMPORTANT: Only include UNREALIZED P&L here. Realized P&L is already in capital.
        if enable_sp500_short and sp500_prices is not None and date in sp500_prices.index and not pd.isna(sp500_prices[date]):
            current_sp500_price_final = sp500_prices[date]
            if sp500_short_shares > 1e-10 and sp500_short_entry_price > 1e-10:
                # Calculate unrealized P&L: profit when price goes down
                # Formula: (entry_price - current_price) * shares
                # If entry=100, current=95, shares=10: P&L = (100-95)*10 = +50 (profit, correct)
                # If entry=100, current=105, shares=10: P&L = (100-105)*10 = -50 (loss, correct)
                sp500_short_pnl_final = sp500_short_shares * (sp500_short_entry_price - current_sp500_price_final)
            else:
                sp500_short_pnl_final = 0.0
        else:
            sp500_short_pnl_final = 0.0
        
        # Recalculate total_value after all rebalancing (stocks and S&P500)
        # Portfolio value = cash + stock holdings + unrealized S&P500 short P&L
        # Note: Realized P&L from S&P500 rebalancing is already included in capital
        # Note: We do NOT double-count - realized P&L is in capital, unrealized P&L is added here
        total_value_final = capital + holdings_value_after_rebalancing + sp500_short_pnl_final
        # Allow negative (margin). Only fallback if non-finite.
        if not np.isfinite(total_value_final):
            total_value_final = INITIAL_CAPITAL  # Fallback
        
        # Track margin usage (when capital is negative)
        if capital < 0:
            margin_days += 1
        
        # Update portfolio value (including S&P500 short P&L)
        portfolio_value.append({
            'date': date,
            'value': total_value_final,
            'cash': capital,
            'holdings_value': holdings_value_after_rebalancing,
            'sp500_short_pnl': sp500_short_pnl_final,
            'sp500_short_shares': sp500_short_shares
        })
        
        # Track target shares and actual shares with rebalance logic
        shares_tracking.append({
            'date': date,
            'target_shares': target_shares.copy(),
            'actual_shares': shares.copy()
        })
        
        # Print progress every 2000 days
        if (i + 1) % PROGRESS_INTERVAL == 0:
            sp500_price_info = ""
            if sp500_prices is not None and date in sp500_prices.index and not pd.isna(sp500_prices[date]):
                sp500_price_info = f", S&P500: ${sp500_prices[date]:,.2f}"
            print(f"Day {i+1}/{len(dates)} ({date.strftime('%Y-%m-%d')}): "
                  f"Portfolio Value: ${total_value_final:,.2f}, "
                  f"Cash: ${capital:,.2f}, "
                  f"Holdings: ${holdings_value_after_rebalancing:,.2f}, "
                  f"S&P500 Short P&L: ${sp500_short_pnl_final:,.2f}, "
                  f"Short Shares: {sp500_short_shares:.2f}{sp500_price_info}")
    
    portfolio_df = pd.DataFrame(portfolio_value)
    shares_tracking_df = pd.DataFrame(shares_tracking)
    
    # Calculate margin % time
    total_days = len(dates)
    margin_pct_time = (margin_days / total_days * 100) if total_days > 0 else 0.0
    
    return portfolio_df, shares, shares_tracking_df, total_commission, margin_pct_time, stop_loss_triggers

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
    """Calculate Maximum Drawdown and return MDD value and period (start, end dates)"""
    if len(values_series) < 2:
        return 0.0, None, None
    
    if isinstance(values_series, pd.Series):
        values = values_series.values
        dates = values_series.index
    else:
        values = np.array(values_series)
        dates = None
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdown
    drawdown = (values - running_max) / running_max
    
    # Maximum drawdown (most negative)
    mdd_idx = drawdown.argmin()  # Index of maximum drawdown (trough)
    mdd = drawdown[mdd_idx] * 100  # Return as percentage
    
    # Find the period: peak before drawdown to trough
    if mdd_idx > 0:
        # Find the peak before the trough
        # The peak is where the running_max was last set before the trough
        # We need to find the index where running_max equals the value at that point
        trough_value = values[mdd_idx]
        peak_value = running_max[mdd_idx]  # The peak value that created this drawdown
        
        # Find the last index where value equals peak_value (before or at mdd_idx)
        peak_idx = mdd_idx
        for i in range(mdd_idx, -1, -1):
            if abs(values[i] - peak_value) < 1e-10 or values[i] >= peak_value:
                peak_idx = i
                break
        
        # The trough is at mdd_idx
        trough_idx = mdd_idx
        
        if dates is not None:
            mdd_start = dates[peak_idx] if peak_idx < len(dates) else None
            mdd_end = dates[trough_idx] if trough_idx < len(dates) else None
        else:
            mdd_start = None
            mdd_end = None
    else:
        mdd_start = None
        mdd_end = None
    
    return abs(mdd), mdd_start, mdd_end  # Return as positive percentage

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

def run_single_backtest(close_df, all_data, period_name=""):
    """Run a single backtest and return results for both with and without S&P500 short"""
    # Calculate VWAP
    vwap = calculate_vwap(close_df, all_data)
    
    # Calculate ranks
    rank_vwap_minus_close, rank_vwap_plus_close = calculate_ranks(vwap, close_df)
    
    # Calculate weights
    weights = calculate_weights(rank_vwap_minus_close, rank_vwap_plus_close)
    
    # Select top 22 stocks and normalize
    top_weights = select_top_stocks(weights, top_n=TOP_N_STOCKS)
    
    # Run backtest WITH S&P500 short
    print(f"\nRunning {period_name} backtest WITH S&P500 short...")
    portfolio_df_short, final_shares_short, shares_tracking_df_short, total_commission_short, margin_pct_time_short, stop_loss_triggers_short = run_backtest(close_df, top_weights, enable_sp500_short=True)
    
    # Run backtest WITHOUT S&P500 short
    print(f"\nRunning {period_name} backtest WITHOUT S&P500 short...")
    portfolio_df_no_short, final_shares_no_short, shares_tracking_df_no_short, total_commission_no_short, margin_pct_time_no_short, stop_loss_triggers_no_short = run_backtest(close_df, top_weights, enable_sp500_short=False)
    
    # Calculate strategy returns for both versions
    initial_value = INITIAL_CAPITAL
    final_value_short = portfolio_df_short['value'].iloc[-1]
    strategy_return_short = (final_value_short / initial_value - 1) * 100
    
    final_value_no_short = portfolio_df_no_short['value'].iloc[-1]
    strategy_return_no_short = (final_value_no_short / initial_value - 1) * 100
    
    # Debug: Print S&P500 short position summary (only for short version)
    if 'sp500_short_pnl' in portfolio_df_short.columns:
        final_short_pnl = portfolio_df_short['sp500_short_pnl'].iloc[-1]
        avg_short_pnl = portfolio_df_short['sp500_short_pnl'].mean()
        final_short_shares = portfolio_df_short['sp500_short_shares'].iloc[-1] if 'sp500_short_shares' in portfolio_df_short.columns else 0
        max_short_shares = portfolio_df_short['sp500_short_shares'].max() if 'sp500_short_shares' in portfolio_df_short.columns else 0
        print(f"\n{period_name} - S&P500 Short Position Summary:")
        print(f"  Final Unrealized P&L: ${final_short_pnl:,.2f}")
        print(f"  Average Unrealized P&L: ${avg_short_pnl:,.2f}")
        print(f"  Final Short Shares: {final_short_shares:.2f}")
        print(f"  Max Short Shares: {max_short_shares:.2f}")
    
    # Calculate performance metrics for both versions
    start_date = close_df.index[0]
    end_date = close_df.index[-1]
    
    # Metrics for version WITH S&P500 short (keep date index for MDD period)
    portfolio_values_short = portfolio_df_short.set_index('date')['value']
    strategy_cagr_short = calculate_cagr(portfolio_values_short, start_date, end_date)
    strategy_sharpe_short = calculate_sharpe_ratio(portfolio_values_short)
    strategy_mdd_short, strategy_mdd_start_short, strategy_mdd_end_short = calculate_mdd(portfolio_values_short)
    strategy_sortino_short = calculate_sortino_ratio(portfolio_values_short)
    
    # Metrics for version WITHOUT S&P500 short
    portfolio_values_no_short = portfolio_df_no_short['value']
    strategy_cagr_no_short = calculate_cagr(portfolio_values_no_short, start_date, end_date)
    strategy_sharpe_no_short = calculate_sharpe_ratio(portfolio_values_no_short)
    strategy_mdd_no_short, _, _ = calculate_mdd(portfolio_values_no_short)  # Don't need period for no_short version
    strategy_sortino_no_short = calculate_sortino_ratio(portfolio_values_no_short)
    
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
        sp500_mdd, _, _ = calculate_mdd(sp500_prices)  # Don't need period for S&P500
        sp500_sortino = calculate_sortino_ratio(sp500_prices)
    
    return {
        'period_name': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'initial_value': initial_value,
        # WITH S&P500 short
        'final_value_short': final_value_short,
        'strategy_return_short': strategy_return_short,
        'total_commission_short': total_commission_short,
        'margin_pct_time_short': margin_pct_time_short,
        'stop_loss_triggers_short': stop_loss_triggers_short,
        'strategy_cagr_short': strategy_cagr_short,
        'strategy_sharpe_short': strategy_sharpe_short,
        'strategy_mdd_short': strategy_mdd_short,
        'strategy_mdd_start_short': strategy_mdd_start_short,
        'strategy_mdd_end_short': strategy_mdd_end_short,
        'strategy_sortino_short': strategy_sortino_short,
        'portfolio_df_short': portfolio_df_short,
        'shares_tracking_df_short': shares_tracking_df_short,
        # WITHOUT S&P500 short
        'final_value_no_short': final_value_no_short,
        'strategy_return_no_short': strategy_return_no_short,
        'total_commission_no_short': total_commission_no_short,
        'margin_pct_time_no_short': margin_pct_time_no_short,
        'stop_loss_triggers_no_short': stop_loss_triggers_no_short,
        'strategy_cagr_no_short': strategy_cagr_no_short,
        'strategy_sharpe_no_short': strategy_sharpe_no_short,
        'strategy_mdd_no_short': strategy_mdd_no_short,
        'strategy_sortino_no_short': strategy_sortino_no_short,
        'portfolio_df_no_short': portfolio_df_no_short,
        'shares_tracking_df_no_short': shares_tracking_df_no_short,
        # S&P500 B&H
        'sp500_cagr': sp500_cagr,
        'sp500_sharpe': sp500_sharpe,
        'sp500_mdd': sp500_mdd,
        'sp500_sortino': sp500_sortino
    }

def main():
    # Load data
    close_df, all_data = load_data()
    
    # Split data into 70% train and 30% test
    total_days = len(close_df)
    train_split_idx = int(total_days * 0.7)
    
    train_close_df = close_df.iloc[:train_split_idx]
    test_close_df = close_df.iloc[train_split_idx:]
    
    print(f"\n{'='*80}")
    print("DATA SPLIT")
    print(f"{'='*80}")
    print(f"Total days: {total_days}")
    print(f"Train period: {train_close_df.index[0]} to {train_close_df.index[-1]} ({len(train_close_df)} days, {len(train_close_df)/total_days*100:.1f}%)")
    print(f"Test period: {test_close_df.index[0]} to {test_close_df.index[-1]} ({len(test_close_df)} days, {len(test_close_df)/total_days*100:.1f}%)")
    
    # Run backtest on train data
    print(f"\n{'='*80}")
    print("Running TRAIN backtest...")
    print(f"{'='*80}")
    train_results = run_single_backtest(train_close_df, all_data, "Train")
    
    # Run backtest on test data
    print(f"\n{'='*80}")
    print("Running TEST backtest...")
    print(f"{'='*80}")
    test_results = run_single_backtest(test_close_df, all_data, "Test")
    
    # Display results side-by-side
    print(f"\n{'='*120}")
    print("BACKTEST RESULTS COMPARISON")
    print(f"{'='*120}")
    print(f"{'Metric':<40} {'Train (No Short)':>20} {'Train (With Short)':>20} {'Test (No Short)':>20} {'Test (With Short)':>20}")
    print(f"{'-'*120}")
    
    # Basic results
    print(f"{'Period':<40} {train_results['start_date'].strftime('%Y-%m-%d'):>20} {'':>20} {test_results['start_date'].strftime('%Y-%m-%d'):>20} {'':>20}")
    print(f"{'  to':<40} {train_results['end_date'].strftime('%Y-%m-%d'):>20} {'':>20} {test_results['end_date'].strftime('%Y-%m-%d'):>20} {'':>20}")
    print(f"{'Initial Capital':<40} ${train_results['initial_value']:>18,.2f} ${train_results['initial_value']:>18,.2f} ${test_results['initial_value']:>18,.2f} ${test_results['initial_value']:>18,.2f}")
    print(f"{'Final Portfolio Value':<40} ${train_results['final_value_no_short']:>18,.2f} ${train_results['final_value_short']:>18,.2f} ${test_results['final_value_no_short']:>18,.2f} ${test_results['final_value_short']:>18,.2f}")
    print(f"{'Total Return (%)':<40} {train_results['strategy_return_no_short']:>19.2f}% {train_results['strategy_return_short']:>19.2f}% {test_results['strategy_return_no_short']:>19.2f}% {test_results['strategy_return_short']:>19.2f}%")
    print(f"{'Total Commission Paid':<40} ${train_results['total_commission_no_short']:>18,.2f} ${train_results['total_commission_short']:>18,.2f} ${test_results['total_commission_no_short']:>18,.2f} ${test_results['total_commission_short']:>18,.2f}")
    print(f"{'Margin % Time':<40} {train_results['margin_pct_time_no_short']:>19.2f}% {train_results['margin_pct_time_short']:>19.2f}% {test_results['margin_pct_time_no_short']:>19.2f}% {test_results['margin_pct_time_short']:>19.2f}%")
    print(f"{'Stop Loss Triggers':<40} {train_results['stop_loss_triggers_no_short']:>20} {train_results['stop_loss_triggers_short']:>20} {test_results['stop_loss_triggers_no_short']:>20} {test_results['stop_loss_triggers_short']:>20}")
    
    print(f"{'-'*120}")
    
    # Performance metrics - Strategy (No Short)
    print(f"{'PERFORMANCE METRICS - STRATEGY (NO SHORT)':<40} {'':>20} {'':>20} {'':>20} {'':>20}")
    print(f"{'CAGR (%)':<40} {train_results['strategy_cagr_no_short']:>19.2f} {'':>20} {test_results['strategy_cagr_no_short']:>19.2f} {'':>20}")
    print(f"{'Sharpe Ratio':<40} {train_results['strategy_sharpe_no_short']:>19.2f} {'':>20} {test_results['strategy_sharpe_no_short']:>19.2f} {'':>20}")
    print(f"{'Max Drawdown (%)':<40} {train_results['strategy_mdd_no_short']:>19.2f} {'':>20} {test_results['strategy_mdd_no_short']:>19.2f} {'':>20}")
    print(f"{'Sortino Ratio':<40} {train_results['strategy_sortino_no_short']:>19.2f} {'':>20} {test_results['strategy_sortino_no_short']:>19.2f} {'':>20}")
    
    print(f"{'-'*120}")
    
    # Performance metrics - Strategy (With Short)
    print(f"{'PERFORMANCE METRICS - STRATEGY (WITH SHORT)':<40} {'':>20} {'':>20} {'':>20} {'':>20}")
    print(f"{'CAGR (%)':<40} {'':>20} {train_results['strategy_cagr_short']:>19.2f} {'':>20} {test_results['strategy_cagr_short']:>19.2f}")
    print(f"{'Sharpe Ratio':<40} {'':>20} {train_results['strategy_sharpe_short']:>19.2f} {'':>20} {test_results['strategy_sharpe_short']:>19.2f}")
    print(f"{'Max Drawdown (%)':<40} {'':>20} {train_results['strategy_mdd_short']:>19.2f} {'':>20} {test_results['strategy_mdd_short']:>19.2f}")
    
    # MDD Period
    train_mdd_start = train_results.get('strategy_mdd_start_short')
    train_mdd_end = train_results.get('strategy_mdd_end_short')
    test_mdd_start = test_results.get('strategy_mdd_start_short')
    test_mdd_end = test_results.get('strategy_mdd_end_short')
    
    def fmt_period(start, end):
        if start is None or end is None:
            return "N/A"
        try:
            return f"{pd.to_datetime(start).strftime('%Y-%m-%d')} to {pd.to_datetime(end).strftime('%Y-%m-%d')}"
        except Exception:
            return "N/A"
    train_mdd_period_str = fmt_period(train_mdd_start, train_mdd_end)
    test_mdd_period_str = fmt_period(test_mdd_start, test_mdd_end)
    
    print(f"{'MDD Period':<40} {'':>20} {train_mdd_period_str:<19} {'':>20} {test_mdd_period_str:<19}")
    print(f"{'Sortino Ratio':<40} {'':>20} {train_results['strategy_sortino_short']:>19.2f} {'':>20} {test_results['strategy_sortino_short']:>19.2f}")
    
    print(f"{'-'*100}")
    
    # Performance metrics - S&P500
    print(f"{'PERFORMANCE METRICS - S&P500 B&H':<30} {'':>20} {'':>20} {'':>20}")
    train_sp500_cagr = train_results['sp500_cagr'] if train_results['sp500_cagr'] is not None else 0.0
    test_sp500_cagr = test_results['sp500_cagr'] if test_results['sp500_cagr'] is not None else 0.0
    train_sp500_cagr_str = f"{train_sp500_cagr:>19.2f}" if train_results['sp500_cagr'] is not None else f"{'N/A':>19}"
    test_sp500_cagr_str = f"{test_sp500_cagr:>19.2f}" if test_results['sp500_cagr'] is not None else f"{'N/A':>19}"
    diff_sp500_cagr = test_sp500_cagr - train_sp500_cagr if (train_results['sp500_cagr'] is not None and test_results['sp500_cagr'] is not None) else None
    diff_sp500_cagr_str = f"{diff_sp500_cagr:>19.2f}" if diff_sp500_cagr is not None else f"{'N/A':>19}"
    print(f"{'CAGR (%)':<30} {train_sp500_cagr_str} {test_sp500_cagr_str} {diff_sp500_cagr_str}")
    
    train_sp500_sharpe_str = f"{train_results['sp500_sharpe']:>19.2f}" if train_results['sp500_sharpe'] is not None else f"{'N/A':>19}"
    test_sp500_sharpe_str = f"{test_results['sp500_sharpe']:>19.2f}" if test_results['sp500_sharpe'] is not None else f"{'N/A':>19}"
    diff_sp500_sharpe = test_results['sp500_sharpe'] - train_results['sp500_sharpe'] if (train_results['sp500_sharpe'] is not None and test_results['sp500_sharpe'] is not None) else None
    diff_sp500_sharpe_str = f"{diff_sp500_sharpe:>19.2f}" if diff_sp500_sharpe is not None else f"{'N/A':>19}"
    print(f"{'Sharpe Ratio':<30} {train_sp500_sharpe_str} {test_sp500_sharpe_str} {diff_sp500_sharpe_str}")
    
    train_sp500_mdd_str = f"{train_results['sp500_mdd']:>19.2f}" if train_results['sp500_mdd'] is not None else f"{'N/A':>19}"
    test_sp500_mdd_str = f"{test_results['sp500_mdd']:>19.2f}" if test_results['sp500_mdd'] is not None else f"{'N/A':>19}"
    diff_sp500_mdd = test_results['sp500_mdd'] - train_results['sp500_mdd'] if (train_results['sp500_mdd'] is not None and test_results['sp500_mdd'] is not None) else None
    diff_sp500_mdd_str = f"{diff_sp500_mdd:>19.2f}" if diff_sp500_mdd is not None else f"{'N/A':>19}"
    print(f"{'Max Drawdown (%)':<30} {train_sp500_mdd_str} {test_sp500_mdd_str} {diff_sp500_mdd_str}")
    
    train_sp500_sortino_str = f"{train_results['sp500_sortino']:>19.2f}" if train_results['sp500_sortino'] is not None else f"{'N/A':>19}"
    test_sp500_sortino_str = f"{test_results['sp500_sortino']:>19.2f}" if test_results['sp500_sortino'] is not None else f"{'N/A':>19}"
    diff_sp500_sortino = test_results['sp500_sortino'] - train_results['sp500_sortino'] if (train_results['sp500_sortino'] is not None and test_results['sp500_sortino'] is not None) else None
    diff_sp500_sortino_str = f"{diff_sp500_sortino:>19.2f}" if diff_sp500_sortino is not None else f"{'N/A':>19}"
    print(f"{'Sortino Ratio':<30} {train_sp500_sortino_str} {test_sp500_sortino_str} {diff_sp500_sortino_str}")
    
    print(f"{'='*120}")
    
    # Save results
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    
    # Save train results (both versions)
    train_portfolio_path_no_short = os.path.join(results_folder, 'portfolio_results_train_no_short.csv')
    train_results['portfolio_df_no_short'].to_csv(train_portfolio_path_no_short, index=False)
    train_portfolio_path_short = os.path.join(results_folder, 'portfolio_results_train_short.csv')
    train_results['portfolio_df_short'].to_csv(train_portfolio_path_short, index=False)
    
    # Save test results (both versions)
    test_portfolio_path_no_short = os.path.join(results_folder, 'portfolio_results_test_no_short.csv')
    test_results['portfolio_df_no_short'].to_csv(test_portfolio_path_no_short, index=False)
    test_portfolio_path_short = os.path.join(results_folder, 'portfolio_results_test_short.csv')
    test_results['portfolio_df_short'].to_csv(test_portfolio_path_short, index=False)
    
    # Save shares tracking for both versions
    for result in [train_results, test_results]:
        period = result['period_name'].lower()
        # Save no_short version
        shares_tracking_df = result['shares_tracking_df_no_short']
        
        target_shares_list = []
        actual_shares_list = []
        
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
        
        target_shares_df.set_index('date', inplace=True)
        actual_shares_df.set_index('date', inplace=True)
        
        target_shares_path = os.path.join(results_folder, f'target_shares_{period}_no_short.csv')
        actual_shares_path = os.path.join(results_folder, f'actual_shares_{period}_no_short.csv')
        actual_shares_transposed_path = os.path.join(results_folder, f'actual_shares_transposed_{period}_no_short.csv')
        
        target_shares_df.to_csv(target_shares_path)
        actual_shares_df.to_csv(actual_shares_path)
        actual_shares_df.T.to_csv(actual_shares_transposed_path)
        
        # Save short version
        shares_tracking_df_short = result['shares_tracking_df_short']
        
        target_shares_list_short = []
        actual_shares_list_short = []
        
        for idx, row in shares_tracking_df_short.iterrows():
            date = row['date']
            target_shares_series = row['target_shares']
            actual_shares_series = row['actual_shares']
            
            target_dict = {'date': date}
            actual_dict = {'date': date}
            
            for stock in target_shares_series.index:
                target_dict[stock] = target_shares_series[stock]
                actual_dict[stock] = actual_shares_series[stock]
            
            target_shares_list_short.append(target_dict)
            actual_shares_list_short.append(actual_dict)
        
        target_shares_df_short = pd.DataFrame(target_shares_list_short)
        actual_shares_df_short = pd.DataFrame(actual_shares_list_short)
        
        target_shares_df_short.set_index('date', inplace=True)
        actual_shares_df_short.set_index('date', inplace=True)
        
        target_shares_path_short = os.path.join(results_folder, f'target_shares_{period}_short.csv')
        actual_shares_path_short = os.path.join(results_folder, f'actual_shares_{period}_short.csv')
        actual_shares_transposed_path_short = os.path.join(results_folder, f'actual_shares_transposed_{period}_short.csv')
        
        target_shares_df_short.to_csv(target_shares_path_short)
        actual_shares_df_short.to_csv(actual_shares_path_short)
        actual_shares_df_short.T.to_csv(actual_shares_transposed_path_short)
    
    print(f"\nResults saved to {results_folder}/ folder")
    print(f"  - portfolio_results_train_no_short.csv, portfolio_results_train_short.csv")
    print(f"  - portfolio_results_test_no_short.csv, portfolio_results_test_short.csv")
    print(f"  - target_shares_train_no_short.csv, actual_shares_train_no_short.csv, actual_shares_transposed_train_no_short.csv")
    print(f"  - target_shares_train_short.csv, actual_shares_train_short.csv, actual_shares_transposed_train_short.csv")
    print(f"  - target_shares_test_no_short.csv, actual_shares_test_no_short.csv, actual_shares_transposed_test_no_short.csv")
    print(f"  - target_shares_test_short.csv, actual_shares_test_short.csv, actual_shares_transposed_test_short.csv")

if __name__ == "__main__":
    main()

