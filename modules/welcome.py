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
        "    [cyan]-s[/cyan], [cyan]--symbols[/cyan] [white](list)[/white]: List of symbols to scan (e.g., !MES, @SPX)"
    )
    console.print(
        f"    [cyan]-p[/cyan], [cyan]--profit[/cyan] [white](float)[/white]: Minimum required ROI profit (default: {default_min_profit})"
    )
    console.print(
        "    [cyan]-l[/cyan], [cyan]--cost-limit[/cyan] [white](float)[/white]: The max cost paid for the option [default: 120]\n"
    )

    console.print(
        "[bold magenta]syn[/bold magenta] - Search for synthetic conversion (synthetic) opportunities not risk free"
    )
    console.print(
        "    [cyan]-s[/cyan], [cyan]--symbols[/cyan] [white](list)[/white]: List of symbols to scan (e.g., !MES, @SPX)"
    )
    console.print(
        "    [cyan]-l[/cyan], [cyan]--cost-limit[/cyan] [white](float)[/white]: Minimum price for the contract [default: 120]"
    )
    console.print(
        "    [cyan]-ml[/cyan], [cyan]--max-loss[/cyan] [white](float)[/white]: Min threshold of the *max loss* for the strategy [default: None]"
    )
    console.print(
        "    [cyan]-mp[/cyan], [cyan]--max-profit[/cyan] [white](float)[/white]: Min threshold of the *max profit* for the strategy [default: None]"
    )
    console.print(
        "    [cyan]-pr[/cyan], [cyan]--profit-ratio[/cyan] [white](float)[/white]: Min threshold of max profit to max loss [max_profit/abs(max_loss)] for the strategy [default: None]\n"
    )

    console.print("[bold underline yellow]Examples:[/bold underline yellow]\n")
    console.print(
        "[green]$ python alchimest.py sfr -s spy qqq -p 0.7 -l 100[/green]  [white]# Scan for SFR with symbols !MES and @SPX, profit >= 0.7, cost limit 100[/white]"
    )
    console.print(
        "[green]$ python alchimest.py syn -s pltr -l 120 -ml -50 -mp 200 -pr 3[/green]  [white]# Synthetic conversion for !MES, cost limit 120, min loss -50, min profit 200, profit ratio 3[/white]\n"
    )
