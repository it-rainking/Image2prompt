#!/usr/bin/env python3
"""
Benchmark Script - Confronto qualità e velocità degli analyzer
Versione: 1.0

Esegue test comparativi su:
1. Moondream 2
2. Florence-2
3. BLIP Base

Output:
- Report JSON con tutti i risultati
- Report TXT leggibile
- Statistiche aggregate
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Import analyzers
from offline_analyzers import (
    create_analyzer,
    AnalyzerType,
    AnalysisResult,
    load_all_datasets
)


# ============================================================================
# CONFIGURAZIONE BENCHMARK
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configurazione del benchmark"""
    output_dir: Path = Path("benchmark_results")
    num_warmup_runs: int = 1  # Run di warmup per caricare modello
    num_timed_runs: int = 3   # Run per calcolare media tempi
    save_individual_results: bool = True


@dataclass
class ImageBenchmark:
    """Risultato benchmark per singola immagine"""
    image_path: str
    image_hash: str
    model_name: str
    base_caption: str
    final_prompt: str
    negative_prompt: str
    inference_times: List[float]
    avg_inference_time: float
    prompt_length: int
    word_count: int


@dataclass
class ModelBenchmark:
    """Risultato aggregato per modello"""
    model_name: str
    total_images: int
    avg_inference_time: float
    min_inference_time: float
    max_inference_time: float
    avg_prompt_length: int
    avg_word_count: int
    load_time: float
    errors: List[str]


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Esegue benchmark comparativi"""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, List[ImageBenchmark]] = {}
        self.model_stats: Dict[str, ModelBenchmark] = {}

    def get_image_hash(self, image_path: str) -> str:
        """Calcola hash MD5 dell'immagine per identificazione"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    def benchmark_single_image(
        self,
        image_path: str,
        analyzer_type: AnalyzerType,
        analyzer=None
    ) -> Optional[ImageBenchmark]:
        """Benchmark singola immagine con singolo modello"""

        if analyzer is None:
            analyzer = create_analyzer(analyzer_type)

        # Warmup run (carica modello se necessario)
        try:
            _ = analyzer.analyze(image_path)
        except Exception as e:
            print(f"  Errore warmup: {e}")
            return None

        # Timed runs
        inference_times = []
        result = None

        for i in range(self.config.num_timed_runs):
            try:
                result = analyzer.analyze(image_path)
                inference_times.append(result.inference_time)
            except Exception as e:
                print(f"  Errore run {i+1}: {e}")
                continue

        if not inference_times or result is None:
            return None

        return ImageBenchmark(
            image_path=str(image_path),
            image_hash=self.get_image_hash(image_path),
            model_name=analyzer_type.value,
            base_caption=result.base_caption,
            final_prompt=result.prompt,
            negative_prompt=result.negative_prompt,
            inference_times=inference_times,
            avg_inference_time=sum(inference_times) / len(inference_times),
            prompt_length=len(result.prompt),
            word_count=len(result.prompt.split())
        )

    def benchmark_model(
        self,
        image_paths: List[str],
        analyzer_type: AnalyzerType
    ) -> Tuple[List[ImageBenchmark], ModelBenchmark]:
        """Benchmark completo per un modello"""

        print(f"\n{'='*60}")
        print(f"BENCHMARK: {analyzer_type.value.upper()}")
        print(f"{'='*60}")

        # Carica modello e misura tempo
        load_start = time.time()
        try:
            analyzer = create_analyzer(analyzer_type)
            # Forza caricamento con dummy analysis
            if image_paths:
                analyzer.analyze(image_paths[0])
        except Exception as e:
            print(f"Errore caricamento modello: {e}")
            return [], ModelBenchmark(
                model_name=analyzer_type.value,
                total_images=0,
                avg_inference_time=0,
                min_inference_time=0,
                max_inference_time=0,
                avg_prompt_length=0,
                avg_word_count=0,
                load_time=0,
                errors=[str(e)]
            )

        load_time = time.time() - load_start
        print(f"Tempo caricamento modello: {load_time:.2f}s")

        # Benchmark ogni immagine
        image_results = []
        errors = []

        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] {Path(img_path).name}")

            try:
                result = self.benchmark_single_image(img_path, analyzer_type, analyzer)
                if result:
                    image_results.append(result)
                    print(f"  Tempo medio: {result.avg_inference_time:.2f}s")
                    print(f"  Parole prompt: {result.word_count}")
                else:
                    errors.append(f"Nessun risultato per {img_path}")
            except Exception as e:
                errors.append(f"{img_path}: {str(e)}")
                print(f"  Errore: {e}")

        # Calcola statistiche aggregate
        if image_results:
            times = [r.avg_inference_time for r in image_results]
            lengths = [r.prompt_length for r in image_results]
            words = [r.word_count for r in image_results]

            model_stats = ModelBenchmark(
                model_name=analyzer_type.value,
                total_images=len(image_results),
                avg_inference_time=sum(times) / len(times),
                min_inference_time=min(times),
                max_inference_time=max(times),
                avg_prompt_length=int(sum(lengths) / len(lengths)),
                avg_word_count=int(sum(words) / len(words)),
                load_time=load_time,
                errors=errors
            )
        else:
            model_stats = ModelBenchmark(
                model_name=analyzer_type.value,
                total_images=0,
                avg_inference_time=0,
                min_inference_time=0,
                max_inference_time=0,
                avg_prompt_length=0,
                avg_word_count=0,
                load_time=load_time,
                errors=errors
            )

        return image_results, model_stats

    def run_full_benchmark(
        self,
        image_paths: List[str],
        models: List[AnalyzerType] = None
    ) -> Dict:
        """Esegue benchmark completo su tutti i modelli"""

        if models is None:
            models = list(AnalyzerType)

        print(f"\n{'#'*60}")
        print(f"BENCHMARK COMPLETO")
        print(f"Immagini: {len(image_paths)}")
        print(f"Modelli: {[m.value for m in models]}")
        print(f"{'#'*60}")

        all_results = {}
        all_stats = {}

        for model_type in models:
            results, stats = self.benchmark_model(image_paths, model_type)
            all_results[model_type.value] = results
            all_stats[model_type.value] = stats

        self.results = all_results
        self.model_stats = all_stats

        return {
            'results': all_results,
            'stats': all_stats
        }

    def save_results(self, output_name: str = None):
        """Salva risultati in JSON e TXT"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = output_name or f"benchmark_{timestamp}"

        # Salva JSON completo
        json_path = self.config.output_dir / f"{output_name}.json"
        json_data = {
            'timestamp': timestamp,
            'config': {
                'num_warmup_runs': self.config.num_warmup_runs,
                'num_timed_runs': self.config.num_timed_runs,
            },
            'model_stats': {k: asdict(v) for k, v in self.model_stats.items()},
            'detailed_results': {
                k: [asdict(r) for r in v]
                for k, v in self.results.items()
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"\nRisultati JSON salvati: {json_path}")

        # Salva report TXT leggibile
        txt_path = self.config.output_dir / f"{output_name}.txt"
        self._save_txt_report(txt_path)
        print(f"Report TXT salvato: {txt_path}")

        return json_path, txt_path

    def _save_txt_report(self, path: Path):
        """Genera report TXT leggibile"""

        lines = []
        lines.append("=" * 70)
        lines.append("BENCHMARK REPORT - OFFLINE IMAGE ANALYZERS")
        lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # Tabella comparativa
        lines.append("\n## CONFRONTO MODELLI\n")
        lines.append(f"{'Modello':<15} {'Tempo Medio':<12} {'Min':<8} {'Max':<8} {'Parole':<8} {'Load':<8}")
        lines.append("-" * 70)

        for model_name, stats in self.model_stats.items():
            lines.append(
                f"{model_name:<15} "
                f"{stats.avg_inference_time:>8.2f}s   "
                f"{stats.min_inference_time:>5.2f}s  "
                f"{stats.max_inference_time:>5.2f}s  "
                f"{stats.avg_word_count:>5}    "
                f"{stats.load_time:>5.2f}s"
            )

        # Dettagli per immagine
        lines.append("\n\n## DETTAGLI PER IMMAGINE\n")

        # Raggruppa per immagine
        images_seen = set()
        for model_name, results in self.results.items():
            for r in results:
                images_seen.add(r.image_path)

        for img_path in images_seen:
            lines.append(f"\n{'─'*70}")
            lines.append(f"IMMAGINE: {Path(img_path).name}")
            lines.append(f"{'─'*70}")

            for model_name, results in self.results.items():
                for r in results:
                    if r.image_path == img_path:
                        lines.append(f"\n[{model_name.upper()}]")
                        lines.append(f"Tempo: {r.avg_inference_time:.2f}s")
                        lines.append(f"Parole: {r.word_count}")
                        lines.append(f"\nCaption base:")
                        lines.append(f"{r.base_caption}")
                        lines.append(f"\nPrompt finale (per SD):")
                        lines.append(f"{r.final_prompt}")
                        lines.append(f"\nNegative prompt:")
                        lines.append(f"{r.negative_prompt}")

        # Errori
        all_errors = []
        for model_name, stats in self.model_stats.items():
            for err in stats.errors:
                all_errors.append(f"[{model_name}] {err}")

        if all_errors:
            lines.append("\n\n## ERRORI\n")
            for err in all_errors:
                lines.append(f"- {err}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def print_summary(self):
        """Stampa riassunto a console"""

        print(f"\n{'='*60}")
        print("RIASSUNTO BENCHMARK")
        print(f"{'='*60}\n")

        print(f"{'Modello':<15} {'Tempo Medio':<12} {'Parole Prompt':<15} {'Caricamento':<12}")
        print("-" * 55)

        for model_name, stats in self.model_stats.items():
            print(
                f"{model_name:<15} "
                f"{stats.avg_inference_time:>8.2f}s   "
                f"{stats.avg_word_count:>8}       "
                f"{stats.load_time:>8.2f}s"
            )

        # Trova migliore per velocità
        if self.model_stats:
            fastest = min(self.model_stats.values(), key=lambda x: x.avg_inference_time if x.avg_inference_time > 0 else float('inf'))
            print(f"\n🏆 Più veloce: {fastest.model_name} ({fastest.avg_inference_time:.2f}s)")


# ============================================================================
# CLI
# ============================================================================

def find_test_images(directory: str = ".") -> List[str]:
    """Trova immagini di test nella directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    images = []

    for ext in extensions:
        images.extend(Path(directory).glob(f"*{ext}"))
        images.extend(Path(directory).glob(f"*{ext.upper()}"))

    return [str(img) for img in images]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Offline Image Analyzers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Benchmark su singola immagine con tutti i modelli
  python benchmark.py -i test.jpg

  # Benchmark su cartella di immagini
  python benchmark.py -d ./test_images

  # Benchmark solo su modelli specifici
  python benchmark.py -i test.jpg -m moondream2 florence2

  # Salva risultati con nome specifico
  python benchmark.py -i test.jpg -o my_benchmark
        """
    )

    parser.add_argument("-i", "--image", help="Path singola immagine")
    parser.add_argument("-d", "--directory", help="Directory con immagini di test")
    parser.add_argument("-m", "--models", nargs="+",
                        choices=["moondream2", "florence2", "blip_base"],
                        help="Modelli da testare (default: tutti)")
    parser.add_argument("-o", "--output", help="Nome file output (senza estensione)")
    parser.add_argument("-r", "--runs", type=int, default=3,
                        help="Numero di run per calcolare media (default: 3)")

    args = parser.parse_args()

    # Trova immagini
    if args.image:
        if not os.path.exists(args.image):
            print(f"Errore: Immagine non trovata: {args.image}")
            sys.exit(1)
        image_paths = [args.image]
    elif args.directory:
        image_paths = find_test_images(args.directory)
        if not image_paths:
            print(f"Errore: Nessuna immagine trovata in {args.directory}")
            sys.exit(1)
    else:
        # Cerca nella directory corrente
        image_paths = find_test_images(".")
        if not image_paths:
            print("Errore: Specifica un'immagine (-i) o una directory (-d)")
            parser.print_help()
            sys.exit(1)

    print(f"Trovate {len(image_paths)} immagini da testare")

    # Configura modelli
    if args.models:
        models = [AnalyzerType(m) for m in args.models]
    else:
        models = list(AnalyzerType)

    # Configura e avvia benchmark
    config = BenchmarkConfig(num_timed_runs=args.runs)
    runner = BenchmarkRunner(config)

    runner.run_full_benchmark(image_paths, models)
    runner.print_summary()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
