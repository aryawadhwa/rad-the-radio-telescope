# Radio Telescope Analyzer
# If you see 'ModuleNotFoundError: No module named "reportlab"', run:
#   pip install reportlab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from datetime import datetime
import os
import glob

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:
    print("\nERROR: The 'reportlab' package is not installed. Please run 'pip install reportlab' and try again.\n")
    exit(1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class DataAnalyzer:
    def __init__(self, data_file=None):
        # Always use the latest cleaned CSV for this run if not specified
        if data_file is None:
            csv_files = glob.glob("radio_data_clean_*.csv")
            if not csv_files:
                print("No cleaned data files found! Run your logger and cleaner first.")
                exit(1)
            self.data_file = max(csv_files, key=os.path.getctime)
            print(f"Using latest data file: {self.data_file}")
        else:
            self.data_file = data_file
        self.data = None
        self.sample_rate = 100  # Hz (from Arduino code)
        self.last_report_text = ''
        
    def load_latest_data(self):
        """Load the most recent data file if none specified"""
        if self.data_file is None:
            data_dir = "radio_telescope_data"
            if not os.path.exists(data_dir):
                print("No data directory found!")
                return False
            
            # Find the most recent file
            csv_files = glob.glob(os.path.join(data_dir, "radio_data_*.csv"))
            if not csv_files:
                print("No data files found!")
                return False
            
            self.data_file = max(csv_files, key=os.path.getctime)
            print(f"Loading latest data file: {self.data_file}")
        
        try:
            self.data = pd.read_csv(self.data_file, header=None, names=['timestamp','raw_adc','voltage','smoothed','baseline_diff','signal_strength'])
            print(f"Loaded {len(self.data)} data points")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def basic_statistics(self):
        """Calculate and display basic statistics"""
        if self.data is None:
            print("No data loaded!")
            return
        
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        stats = self.data[['raw_adc', 'voltage', 'smoothed', 'baseline_diff', 'signal_strength']].describe()
        print(stats)
        
        # Duration of observation
        duration = (self.data['timestamp'].max() - self.data['timestamp'].min()) / 1000.0
        print(f"\nObservation Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Data Points: {len(self.data)}")
        print(f"Average Sample Rate: {len(self.data)/duration:.1f} Hz")
    
    def detect_signals(self, threshold_factor=3):
        """Detect potential radio signals"""
        if self.data is None:
            return
        
        # Calculate detection threshold
        mean_strength = self.data['signal_strength'].mean()
        std_strength = self.data['signal_strength'].std()
        threshold = mean_strength + (threshold_factor * std_strength)
        
        # Find signals above threshold
        signals = self.data[self.data['signal_strength'] > threshold].copy()
        
        print(f"\n" + "="*50)
        print("SIGNAL DETECTION")
        print("="*50)
        print(f"Detection Threshold: {threshold:.2f}")
        print(f"Number of signal events: {len(signals)}")
        
        if len(signals) > 0:
            print(f"Strongest signal: {signals['signal_strength'].max():.2f}")
            print(f"Average signal strength: {signals['signal_strength'].mean():.2f}")
            
            # Show time periods with signals
            print("\nSignal Events:")
            for idx, row in signals.head(10).iterrows():
                time_sec = row['timestamp'] / 1000.0
                print(f"  Time: {time_sec:.1f}s, Strength: {row['signal_strength']:.2f}")
    
    def frequency_analysis(self):
        """Perform frequency domain analysis"""
        if self.data is None:
            return
        
        # Use signal strength for frequency analysis
        signal_data = self.data['signal_strength'].values
        
        # Remove DC component
        signal_data = signal_data - np.mean(signal_data)
        
        # Apply window function
        windowed_signal = signal_data * np.hanning(len(signal_data))
        
        # Compute FFT
        fft_result = fft(windowed_signal)
        frequencies = fftfreq(len(signal_data), 1/self.sample_rate)
        
        # Only positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        fft_magnitude = np.abs(fft_result[positive_freq_idx])
        
        # Find peaks
        peaks, properties = signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)
        
        print(f"\n" + "="*50)
        print("FREQUENCY ANALYSIS")
        print("="*50)
        print(f"Frequency range: 0 to {self.sample_rate/2} Hz")
        
        if len(peaks) > 0:
            print("Significant frequency components:")
            for peak in peaks[:5]:  # Show top 5 peaks
                freq = frequencies[peak]
                magnitude = fft_magnitude[peak]
                print(f"  {freq:.2f} Hz: {magnitude:.2f}")
        
        return frequencies, fft_magnitude, peaks
    
    def create_comprehensive_plots(self):
        """Create comprehensive analysis plots and save as PDF (A4)"""
        if self.data is None:
            print("No data loaded!")
            return
        
        # Create figure with subplots (A4 size in inches: 8.27 x 11.69)
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Convert timestamp to time in minutes
        time_minutes = (self.data['timestamp'] - self.data['timestamp'].min()) / 60000.0
        
        # 1. Raw signal over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_minutes, self.data['raw_adc'], 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_title('Raw ADC Signal Over Time')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('ADC Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Smoothed signal
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_minutes, self.data['smoothed'], 'g-', linewidth=1)
        ax2.set_title('Smoothed Signal')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Smoothed Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Baseline difference
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(time_minutes, self.data['baseline_diff'], 'r-', linewidth=1)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Baseline Difference')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Difference')
        ax3.grid(True, alpha=0.3)
        
        # 4. Signal strength
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(time_minutes, self.data['signal_strength'], 'm-', linewidth=1)
        ax4.set_title('Signal Strength')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Strength')
        ax4.grid(True, alpha=0.3)
        
        # 5. Signal strength histogram
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(self.data['signal_strength'], bins=50, alpha=0.7, color='purple')
        ax5.set_title('Signal Strength Distribution')
        ax5.set_xlabel('Signal Strength')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Frequency analysis
        frequencies, fft_magnitude, peaks = self.frequency_analysis()
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(frequencies, fft_magnitude, 'b-', linewidth=1)
        if len(peaks) > 0:
            ax6.plot(frequencies[peaks], fft_magnitude[peaks], 'ro', markersize=8)
        ax6.set_title('Frequency Spectrum')
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Magnitude')
        ax6.set_xlim(0, 10)  # Focus on low frequencies
        ax6.grid(True, alpha=0.3)
        
        # 7. Signal correlation plot
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.scatter(self.data['smoothed'], self.data['signal_strength'], alpha=0.6, s=1)
        ax7.set_title('Smoothed vs Signal Strength')
        ax7.set_xlabel('Smoothed Value')
        ax7.set_ylabel('Signal Strength')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Radio Telescope Data Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save plot as PDF (A4)
        plot_pdf_filename = f"radio_data_clean_analysis_{timestamp}.pdf"
        with PdfPages(plot_pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"Analysis plot saved as PDF: {plot_pdf_filename}")
    
    def generate_report(self):
        """Generate a comprehensive analysis report and save as PDF (A4)"""
        if self.data is None:
            print("No data loaded!")
            return
        
        report_filename = f"radio_data_clean_report_{timestamp}.txt"
        pdf_report_filename = f"radio_data_clean_report_{timestamp}.pdf"
        
        # Generate the text report as before
        report_lines = []
        report_lines.append("RADIO TELESCOPE DATA ANALYSIS REPORT\n")
        report_lines.append("=" * 50 + "\n\n")
        report_lines.append(f"Data File: {self.data_file}\n")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        duration = (self.data['timestamp'].max() - self.data['timestamp'].min()) / 1000.0
        report_lines.append(f"Observation Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n")
        report_lines.append(f"Total Data Points: {len(self.data)}\n")
        report_lines.append(f"Average Sample Rate: {len(self.data)/duration:.1f} Hz\n\n")
        stats = self.data[['raw_adc', 'voltage', 'smoothed', 'baseline_diff', 'signal_strength']].describe()
        report_lines.append("SIGNAL STATISTICS:\n")
        report_lines.append(str(stats))
        report_lines.append("\n\n")
        mean_strength = self.data['signal_strength'].mean()
        std_strength = self.data['signal_strength'].std()
        threshold = mean_strength + (3 * std_strength)
        signals = self.data[self.data['signal_strength'] > threshold]
        report_lines.append(f"SIGNAL DETECTION (3-sigma threshold):\n")
        report_lines.append(f"Detection Threshold: {threshold:.2f}\n")
        report_lines.append(f"Number of signal events: {len(signals)}\n")
        if len(signals) > 0:
            report_lines.append(f"Strongest signal: {signals['signal_strength'].max():.2f}\n")
            report_lines.append(f"Average signal strength: {signals['signal_strength'].mean():.2f}\n")
        report_lines.append("\n")
        report_lines.append("RECOMMENDATIONS:\n")
        report_lines.append("1. Point antenna at Sun during day for strong radio source\n")
        report_lines.append("2. Try observations at different times of day\n")
        report_lines.append("3. Look for periodic variations in signal strength\n")
        report_lines.append("4. Compare signal levels when pointing at different sky regions\n")
        report_lines.append("5. Record weather conditions and correlate with signal variations\n")
        report_text = ''.join(report_lines)
        self.last_report_text = report_text
        # Save as TXT
        with open(report_filename, 'w') as f:
            f.write(report_text)
        print(f"Analysis report saved as: {report_filename}")
        # Save as PDF (A4)
        c = canvas.Canvas(pdf_report_filename, pagesize=A4)
        width, height = A4
        margin = 40
        y = height - margin
        for line in report_text.split('\n'):
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, line)
            y -= 14  # line height
        c.save()
        print(f"Analysis report saved as PDF: {pdf_report_filename}")

def main():
    print("Radio Telescope Data Analyzer")
    print("=" * 40)
    
    # Create analyzer
    analyzer = DataAnalyzer()
    
    # Load data
    if not analyzer.load_latest_data():
        return
    
    # Perform analysis
    analyzer.basic_statistics()
    analyzer.detect_signals()
    analyzer.frequency_analysis()
    
    # Create plots
    analyzer.create_comprehensive_plots()
    
    # Generate report
    analyzer.generate_report()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 