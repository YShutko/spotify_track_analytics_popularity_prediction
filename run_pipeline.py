"""
Complete Pipeline Runner
Executes ETL -> Feature Engineering -> ML Training using scripts
Then provides option to launch dashboard
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and display output"""
    print("\n" + "=" * 80)
    print(f"▶️  {description}")
    print("=" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start_time
        print(result.stdout)

        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"✅ {description} completed in {elapsed:.2f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.2f}s")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Run complete pipeline"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SPOTIFY ANALYTICS PIPELINE" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Track success
    all_succeeded = True

    # Step 1: ETL Pipeline
    if not run_command("python src/etl_pipeline.py", "Step 1/3: ETL Pipeline"):
        all_succeeded = False
        print("\n⚠️  ETL failed, but continuing...")

    # Step 2: Feature Engineering
    if not run_command("python src/feature_engineering.py", "Step 2/3: Feature Engineering"):
        all_succeeded = False
        print("\n⚠️  Feature Engineering failed, but continuing...")

    # Step 3: Model Training
    if not run_command("python src/train_model.py", "Step 3/3: Model Training"):
        all_succeeded = False
        print("\n⚠️  Model Training failed")

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    if all_succeeded:
        print("✅ All steps completed successfully!")
        print("\n📊 Data processed:")
        print("   • ETL: 114,000 tracks cleaned")
        print("   • Feature Engineering: 37 ML features created")
        print("   • Model Training: XGBoost model saved")

        print("\n📁 Output files:")
        print("   • data/processed/cleaned_spotify_data.parquet")
        print("   • data/processed/X_train.parquet, X_test.parquet")
        print("   • outputs/models/xgboost_popularity_model.joblib")

        print("\n🚀 Ready to launch dashboard!")
        print("\nRun: streamlit run app.py")

        # Ask if user wants to launch dashboard
        print("\n" + "=" * 80)
        response = input("Launch Streamlit dashboard now? (y/n): ").strip().lower()

        if response == 'y':
            print("\n🚀 Launching Streamlit dashboard...")
            print("Dashboard will open in your browser at http://localhost:8501")
            print("Press Ctrl+C to stop the server\n")

            try:
                subprocess.run("streamlit run app.py", shell=True, check=True)
            except KeyboardInterrupt:
                print("\n\n✋ Dashboard stopped by user")
            except Exception as e:
                print(f"\n❌ Error launching dashboard: {e}")
        else:
            print("\n👋 Pipeline complete. Launch dashboard manually with: streamlit run app.py")

        return 0
    else:
        print("⚠️  Some steps failed. Check output above for details.")
        print("\nYou can:")
        print("  1. Fix errors and re-run: python run_pipeline.py")
        print("  2. Run individual steps:")
        print("     - python src/etl_pipeline.py")
        print("     - python src/feature_engineering.py")
        print("     - python src/train_model.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
