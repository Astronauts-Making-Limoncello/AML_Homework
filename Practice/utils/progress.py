from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

def get_progress_bar():
  
  return Progress(
    TextColumn("[progress.description]{task.description}"),
    TextColumn("[progress.percentage]{task.percentage:>3.2f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    TextColumn("[#00008B]{task.speed} it/s"),
    SpinnerColumn()
  )