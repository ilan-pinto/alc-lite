def print_welcome(console, version, default_min_profit):
    """
    Prints the welcome message, colored argument summary, and usage examples.
    Args:
        console: rich.console.Console instance
        version: str, version string
        default_min_profit: float, default minimum profit
    """
    import pyfiglet

    welcome_message = pyfiglet.figlet_format("alc-lite")
    console.print(f"[bold green]{welcome_message}[/bold green]")
    console.print(f"[bold cyan]Version: {version}[/bold cyan]\n")

    # Colored summary of arguments
    console.print(
        "[bold underline yellow]Available Commands & Arguments:[/bold underline yellow]\n"
    )
    console.print(
        "[bold magenta]sfr[/bold magenta] - Search for synthetic risk free arbitrage opportunities"
    )
    console.print(
        "    [cyan]-s[/cyan], [cyan]--symbols[/cyan] [white](list)[/white]: List of symbols to scan (e.g., SPY, QQQ)"
    )
    console.print(
        f"    [cyan]-p[/cyan], [cyan]--profit[/cyan] [white](float)[/white]: Minimum required ROI profit (default: {default_min_profit})"
    )
    console.print(
        "    [cyan]-l[/cyan], [cyan]--cost-limit[/cyan] [white](float)[/white]: The max cost paid for the option [default: 120]"
    )
    console.print(
        "    [cyan]-q[/cyan], [cyan]--quantity[/cyan] [white](int)[/white]: Quantity of the option"
    )
    console.print(
        "    [cyan]-ml[/cyan], [cyan]--max-loss[/cyan] [white](float)[/white]: Min threshold of the *max loss* for the strategy [default: None]"
    )
    console.print(
        "    [cyan]-mp[/cyan], [cyan]--max-profit[/cyan] [white](float)[/white]: Min threshold of the *max profit* for the strategy [default: None]"
    )
    console.print(
        "    [cyan]-pr[/cyan], [cyan]--profit-ratio[/cyan] [white](float)[/white]: Min threshold of max profit to max loss [max_profit/abs(max_loss)] for the strategy [default: None]"
    )
    console.print(
        "    [cyan]--debug[/cyan]: Enable debug logging (shows all log levels: DEBUG, INFO, WARNING, ERROR)"
    )
    console.print(
        "    [cyan]--warning[/cyan]: Enable warning logging (shows INFO and WARNING levels only)"
    )
    console.print(
        "    [cyan]--log[/cyan] [white](file)[/white]: Log file path to write all logs to a text file\n"
    )

    console.print(
        "[bold magenta]syn[/bold magenta] - Search for synthetic (non-risk-free) arbitrage opportunities"
    )
    console.print(
        "    [cyan]--scoring-strategy[/cyan] [white](choice)[/white]: Pre-defined scoring strategy for global opportunity selection [default: balanced]"
    )
    console.print(
        "        [dim]• conservative: Prioritizes safety and liquidity (30% risk-reward, 35% liquidity)[/dim]"
    )
    console.print(
        "        [dim]• aggressive: Prioritizes maximum returns (50% risk-reward, 15% liquidity)[/dim]"
    )
    console.print(
        "        [dim]• balanced: Balanced approach between risk and return (35% risk-reward, 25% liquidity)[/dim]"
    )
    console.print(
        "        [dim]• liquidity-focused: Prioritizes highly liquid options (20% risk-reward, 40% liquidity)[/dim]"
    )
    console.print(
        "    [cyan]--risk-reward-weight[/cyan] [white](float)[/white]: Custom weight for risk-reward ratio (overrides strategy)"
    )
    console.print(
        "    [cyan]--liquidity-weight[/cyan] [white](float)[/white]: Custom weight for liquidity score (overrides strategy)"
    )
    console.print(
        "    [dim]All SFR options above are also available for syn command[/dim]\n"
    )

    console.print(
        "[bold magenta]calendar[/bold magenta] - Search for calendar spread arbitrage opportunities"
    )
    console.print(
        "    [cyan]-s[/cyan], [cyan]--symbols[/cyan] [white](list)[/white]: List of symbols to scan for calendar spreads"
    )
    console.print(
        "    [cyan]-l[/cyan], [cyan]--cost-limit[/cyan] [white](float)[/white]: Maximum net debit to pay for calendar spread [default: 300]"
    )
    console.print(
        "    [cyan]-p[/cyan], [cyan]--profit-target[/cyan] [white](float)[/white]: Target profit as percentage of max profit [default: 0.25 = 25%]"
    )
    console.print(
        "    [cyan]--iv-spread-threshold[/cyan] [white](float)[/white]: Minimum IV spread (back - front) required [default: 0.03 = 3%]"
    )
    console.print(
        "    [cyan]--theta-ratio-threshold[/cyan] [white](float)[/white]: Minimum theta ratio (front/back) required [default: 1.5]"
    )
    console.print(
        "    [cyan]--front-expiry-max-days[/cyan] [white](int)[/white]: Maximum days to expiry for front month [default: 45]"
    )
    console.print(
        "    [cyan]--back-expiry-min-days[/cyan] [white](int)[/white]: Minimum days to expiry for back month [default: 50]"
    )
    console.print(
        "    [cyan]--back-expiry-max-days[/cyan] [white](int)[/white]: Maximum days to expiry for back month [default: 120]"
    )
    console.print(
        "    [cyan]--min-volume[/cyan] [white](int)[/white]: Minimum daily volume per option leg [default: 10]"
    )
    console.print(
        "    [cyan]--max-bid-ask-spread[/cyan] [white](float)[/white]: Maximum bid-ask spread as percent of mid price [default: 0.15 = 15%]"
    )
    console.print(
        "    [dim]Calendar spreads profit from time decay differential between front and back month options[/dim]\n"
    )

    console.print("[bold underline yellow]Examples:[/bold underline yellow]\n")
    console.print(
        "[green]$ alc-lite sfr --symbols SPY QQQ --cost-limit 100 --profit 0.75[/green]  [white]# Basic SFR scan with cost limit $100 and min profit 0.75%[/white]"
    )
    console.print(
        "[green]$ alc-lite sfr --warning --symbols TSLA --cost-limit 100[/green]  [white]# SFR scan with warning-level logging (INFO + WARNING messages)[/white]"
    )
    console.print(
        "[green]$ alc-lite sfr --debug --log trades.log --symbols NVDA[/green]  [white]# SFR scan with debug logging to both console and file[/white]"
    )
    console.print(
        "[green]$ alc-lite syn --symbols AAPL MSFT GOOGL --scoring-strategy balanced[/green]  [white]# Synthetic scan with global selection using balanced strategy[/white]"
    )
    console.print(
        "[green]$ alc-lite syn --symbols SPY QQQ IWM --scoring-strategy conservative[/green]  [white]# Conservative strategy: prioritizes safety and liquidity[/white]"
    )
    console.print(
        "[green]$ alc-lite syn --symbols TSLA NVDA --scoring-strategy aggressive[/green]  [white]# Aggressive strategy: maximizes returns with higher risk[/white]"
    )
    console.print(
        "[green]$ alc-lite syn --symbols SPY QQQ --risk-reward-weight 0.5 --liquidity-weight 0.3[/green]  [white]# Custom scoring weights for fine-tuned selection[/white]"
    )
    console.print(
        "[green]$ alc-lite calendar --symbols SPY QQQ AAPL --cost-limit 300 --profit-target 0.25[/green]  [white]# Basic calendar spread scan with $300 cost limit and 25% profit target[/white]"
    )
    console.print(
        "[green]$ alc-lite calendar --symbols AAPL TSLA --iv-spread-threshold 0.04 --theta-ratio-threshold 2.0[/green]  [white]# High IV environment calendar spreads[/white]"
    )
    console.print(
        "[green]$ alc-lite calendar --symbols SPY IWM --front-expiry-max-days 30 --min-volume 50[/green]  [white]# Conservative calendar parameters with higher volume requirement[/white]\n"
    )
