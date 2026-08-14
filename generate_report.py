import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

from backtester import run_backtest
from data_loader import load_data
from features import build_feature_matrix


class Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def create_trade_chart(df_test, res, output_img="trades_chart.png"):
    """Erstellt einen hochauflösenden Chart mit Einstiegen, SL/TP-Linien und Equity Curve."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

                               
    ax1.plot(df_test.index, df_test['close'], label='AAPL Close', color='#2c3e50', alpha=0.6, linewidth=1)

                        
    for trade in res.trades:
        entry_time = trade.entry_time
        exit_time = trade.exit_time if trade.exit_time is not None else df_test.index[-1]
        
        entry_price = trade.entry_price
        exit_price = trade.exit_price if trade.exit_price is not None else entry_price
        sl_price = trade.sl_price
        tp_price = trade.tp_price

                                       
        ax1.scatter(entry_time, entry_price, color='green', marker='^', s=80, zorder=5)

                                               
        ax1.plot([entry_time, exit_time], [sl_price, sl_price], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.plot([entry_time, exit_time], [tp_price, tp_price], color='green', linestyle='--', linewidth=1.2, alpha=0.7)

                                                                                                
        is_profit = exit_price > entry_price
        line_color = '#27ae60' if is_profit else '#e74c3c'
        ax1.plot([entry_time, exit_time], [entry_price, exit_price], color=line_color, linewidth=1.5, alpha=0.8)

    ax1.set_title("AAPL 15m — Trades der letzten 30 Tage (Grün: Entry/TP, Rot: SL)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Preis ($)", fontsize=11)
    ax1.grid(True, alpha=0.3)

                                        
    equity_series = pd.Series(res.equity_curve, index=df_test.index[:len(res.equity_curve)])
    ax2.plot(equity_series.index, equity_series.values, color='#2980b9', linewidth=1.8, label='Equity ($)')
    ax2.set_title("Equity Curve ($)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Portfolio ($)", fontsize=11)
    ax2.grid(True, alpha=0.3)

                                        
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()


def generate_pdf_report(res, chart_img, output_pdf="Trial_3055_Report.pdf"):
    """Erstellt das PDF-Dokument mit Statistiken und dem generierten Chart."""
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor("#2c3e50"))
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor("#34495e"))

    story = []

           
    story.append(Paragraph("<b>Trading-Strategie Report: Trial #3055</b>", title_style))
    story.append(Paragraph("Out-of-Sample Backtest — Letzte 30 Handelstage (15m Timeframe)", h2_style))
    story.append(Spacer(1, 15))

                        
    table_data = [
        ["Metrik", "Wert", "Metrik", "Wert"],
        ["Total Return", f"{res.total_return_pct * 100:+.2f}%", "Winrate", f"{res.winrate * 100:.1f}%"],
        ["Anzahl Trades", f"{res.n_trades}", "Profit Factor", f"{res.profit_factor:.2f}"],
        ["Max Drawdown", f"{res.max_drawdown_pct * 100:.2f}%", "Ø CRV", f"{res.avg_crv:.2f}"],
    ]

    t = Table(table_data, colWidths=[130, 120, 130, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 20))

                    
    story.append(Paragraph("<b>Visuelle Trade-Analyse & Kontoverlauf</b>", h2_style))
    story.append(Spacer(1, 8))
    story.append(Image(chart_img, width=540, height=380))

                  
    doc.build(story)
    print(f"PDF-Report erfolgreich erstellt: {output_pdf}")


def main():
    if not os.path.exists("aapl_3055_model.pkl") or not os.path.exists("aapl_3055_config.pkl"):
        print("Model- oder Config-Datei nicht gefunden! Bitte zuerst export_model.py ausführen.")
        return

    model = joblib.load("aapl_3055_model.pkl")
    config = joblib.load("aapl_3055_config.pkl")
    trial_params = Params(config['params'])
    feature_cols = config['feature_cols']

    print("Lade AAPL 15m Daten...")
    df_raw = load_data(ticker="AAPL")
    df = build_feature_matrix(df_raw, trial_params)

    split_idx = int(len(df) * 0.5)
    df_test = df.iloc[split_idx:].copy()

    X_test = df_test[feature_cols]
    probs = model.predict_proba(X_test)[:, 1]

    base_confidences = pd.Series(probs, index=df_test.index)
    base_signals = (base_confidences >= trial_params.signal_threshold).astype(int)

    times = pd.to_datetime(df_test.index).time
    no_trade_mask = times >= pd.to_datetime("21:00").time()
    base_signals[no_trade_mask] = 0

    print("Führe Backtest aus...")
    res = run_backtest(
        df=df_test,
        signals=base_signals,
        confidences=base_confidences,
        sl_atr_mult=trial_params.sl_atr_mult,
        tp_atr_mult=trial_params.tp_atr_mult,
        max_hold_bars=999,
        force_eod_close=True,
    )

    chart_file = "trades_chart.png"
    pdf_file = "Trial_3055_Report.pdf"

    print("Erstelle Trade-Chart...")
    create_trade_chart(df_test, res, output_img=chart_file)

    print("Generiere PDF-Report...")
    generate_pdf_report(res, chart_file, output_pdf=pdf_file)


if __name__ == "__main__":
    main()