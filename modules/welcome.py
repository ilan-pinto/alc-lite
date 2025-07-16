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

    console.print("[bold underline yellow]Examples:[/bold underline yellow]\n")
    console.print(
        "[green]$ alc-lite sfr -s spy qqq -p 0.7 -l 100 -q 2[/green]  [white]# Scan for SFR with symbols SPY and QQQ, profit >= 0.7, cost limit 100, quantity 2[/white]"
    )
    console.print(
        "[green]$ alc-lite syn -s PLTR -l 120 -ml -50 -mp 200 -pr 0.3 -q 3[/green]  [white]# Synthetic conversion for PLTR, cost limit 120, min loss -50, min profit 200, profit ratio 0.3, quantity 3[/white]"
    )
    console.print(
        "[green]$ alc-lite sfr --warning -s TSLA -l 100[/green]  [white]# SFR scan with warning-level logging (INFO + WARNING messages)[/white]"
    )
    console.print(
        "[green]$ alc-lite syn --debug -s NVDA --log debug.txt[/green]  [white]# Synthetic scan with debug logging to console and file[/white]\n"
    )
