#!/usr/bin/env python3
"""
Offline Image Analyzers - Generatori di prompt SD senza API
Versione: 1.0

Tre implementazioni per benchmark:
1. Moondream 2 - Migliore qualità/velocità (4-6GB RAM)
2. Florence-2 - Più leggero (1-2GB RAM)
3. BLIP Base - Già nel repository (2-3GB RAM)

Tutti funzionano:
- Gratis (open source)
- Offline (dopo primo download)
- Senza GPU (solo CPU)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATA_PATH = Path(__file__).parent / "clip_interrogator" / "data"

@dataclass
class AnalysisResult:
    """Risultato dell'analisi immagine"""
    prompt: str
    negative_prompt: str
    base_caption: str
    model_name: str
    inference_time: float  # secondi
    metadata: Optional[Dict] = None


class AnalyzerType(Enum):
    MOONDREAM2 = "moondream2"
    FLORENCE2 = "florence2"
    BLIP_BASE = "blip_base"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_list(data_path: Path, filename: str) -> List[str]:
    """Carica lista da file di testo"""
    filepath = data_path / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []


def load_all_datasets(data_path: Path = DATA_PATH) -> Dict[str, List[str]]:
    """Carica tutti i dataset per arricchimento prompt"""
    return {
        'flavors': load_list(data_path, 'flavors.txt'),
        'mediums': load_list(data_path, 'mediums.txt'),
        'movements': load_list(data_path, 'movements.txt'),
        'artists': load_list(data_path, 'artists.txt'),
        'negative': load_list(data_path, 'negative.txt'),
    }


# ============================================================================
# BASE ANALYZER CLASS
# ============================================================================

class BaseOfflineAnalyzer(ABC):
    """Classe base per tutti gli analyzer offline"""

    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.datasets = load_all_datasets(data_path)
        self.model = None
        self.model_name = "base"
        self._is_loaded = False

    @abstractmethod
    def load_model(self):
        """Carica il modello in memoria"""
        pass

    @abstractmethod
    def generate_caption(self, image_path: str) -> str:
        """Genera caption base dall'immagine"""
        pass

    def enhance_prompt(self, base_caption: str) -> str:
        """Arricchisce il prompt con termini SD ottimizzati"""
        enhanced = base_caption.strip()

        # Quality terms da aggiungere se non presenti
        quality_terms = [
            'highly detailed',
            'sharp focus',
            '8k',
            'professional',
        ]

        caption_lower = enhanced.lower()
        for term in quality_terms:
            if term.lower() not in caption_lower:
                enhanced += f", {term}"

        return enhanced

    def generate_negative(self, count: int = 15) -> str:
        """Genera negative prompt standard"""
        negatives = self.datasets.get('negative', [])
        if negatives:
            return ", ".join(negatives[:count])
        # Fallback se file non trovato
        return "blurry, bad anatomy, bad hands, cropped, worst quality, low quality, normal quality, jpeg artifacts"

    def analyze(self, image_path: str) -> AnalysisResult:
        """Analisi completa dell'immagine"""
        if not self._is_loaded:
            print(f"[{self.model_name}] Caricamento modello...")
            load_start = time.time()
            self.load_model()
            load_time = time.time() - load_start
            print(f"[{self.model_name}] Modello caricato in {load_time:.2f}s")
            self._is_loaded = True

        # Step 1: Caption base
        start_time = time.time()
        caption = self.generate_caption(image_path)

        # Step 2: Arricchimento
        prompt = self.enhance_prompt(caption)

        # Step 3: Negative
        negative = self.generate_negative()

        inference_time = time.time() - start_time

        return AnalysisResult(
            prompt=prompt,
            negative_prompt=negative,
            base_caption=caption,
            model_name=self.model_name,
            inference_time=inference_time
        )


# ============================================================================
# MOONDREAM 2 ANALYZER
# ============================================================================

class Moondream2Analyzer(BaseOfflineAnalyzer):
    """
    Analizzatore con Moondream 2
    - RAM: 4-6GB
    - Tempo CPU: 5-10s per immagine
    - Qualità: Buona per VQA generale
    """

    def __init__(self, data_path: Path = DATA_PATH):
        super().__init__(data_path)
        self.model_name = "moondream2"
        self.tokenizer = None

    def load_model(self):
        """Carica Moondream 2"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_id = "vikhyatk/moondream2"
            revision = "2024-08-26"  # Versione stabile

            print(f"[{self.model_name}] Scaricamento/caricamento da {model_id}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPU usa float32
                device_map='cpu',
                low_cpu_mem_usage=True
            )
            self.model.eval()

        except ImportError as e:
            raise ImportError(
                f"Dipendenze mancanti per Moondream 2: {e}\n"
                "Installa con: pip install transformers torch accelerate"
            )

    def generate_caption(self, image_path: str) -> str:
        """Genera caption con Moondream 2"""
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        # Moondream ha un metodo specifico per encoding
        enc_image = self.model.encode_image(image)

        # Prompt ottimizzato per Stable Diffusion
        question = (
            "Describe this image in detail for an AI image generator. "
            "Include the main subject, artistic style, lighting, colors, "
            "mood, composition, and any notable details. "
            "Be specific and descriptive."
        )

        answer = self.model.answer_question(enc_image, question, self.tokenizer)

        return answer.strip()


# ============================================================================
# FLORENCE-2 ANALYZER
# ============================================================================

class Florence2Analyzer(BaseOfflineAnalyzer):
    """
    Analizzatore con Florence-2 Base
    - RAM: 1-2GB
    - Tempo CPU: 3-8s per immagine
    - Qualità: Buona, multi-task
    """

    def __init__(self, data_path: Path = DATA_PATH):
        super().__init__(data_path)
        self.model_name = "florence2"
        self.processor = None

    def load_model(self):
        """Carica Florence-2"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch

            model_id = "microsoft/Florence-2-base"

            print(f"[{self.model_name}] Scaricamento/caricamento da {model_id}...")

            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.model.eval()

        except ImportError as e:
            raise ImportError(
                f"Dipendenze mancanti per Florence-2: {e}\n"
                "Installa con: pip install transformers torch"
            )

    def generate_caption(self, image_path: str) -> str:
        """Genera caption con Florence-2"""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert('RGB')

        # Florence-2 usa task prompts speciali
        # <MORE_DETAILED_CAPTION> per descrizioni dettagliate
        task_prompt = "<MORE_DETAILED_CAPTION>"

        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Rimuovi il task prompt dalla risposta se presente
        if generated_text.startswith(task_prompt):
            generated_text = generated_text[len(task_prompt):].strip()

        return generated_text.strip()


# ============================================================================
# BLIP BASE ANALYZER
# ============================================================================

class BLIPBaseAnalyzer(BaseOfflineAnalyzer):
    """
    Analizzatore con BLIP Base (già nel repository)
    - RAM: 2-3GB
    - Tempo CPU: 10-30s per immagine
    - Qualità: Discreta per captioning
    """

    def __init__(self, data_path: Path = DATA_PATH):
        super().__init__(data_path)
        self.model_name = "blip_base"
        self.processor = None

    def load_model(self):
        """Carica BLIP Base"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            model_id = "Salesforce/blip-image-captioning-base"

            print(f"[{self.model_name}] Scaricamento/caricamento da {model_id}...")

            self.processor = BlipProcessor.from_pretrained(model_id)

            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
            self.model.eval()

        except ImportError as e:
            raise ImportError(
                f"Dipendenze mancanti per BLIP: {e}\n"
                "Installa con: pip install transformers torch"
            )

    def generate_caption(self, image_path: str) -> str:
        """Genera caption con BLIP Base"""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert('RGB')

        # Conditional captioning con prompt
        text = "a photograph of"

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                do_sample=False
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption.strip()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_analyzer(analyzer_type: AnalyzerType, data_path: Path = DATA_PATH) -> BaseOfflineAnalyzer:
    """Factory per creare l'analyzer appropriato"""
    analyzers = {
        AnalyzerType.MOONDREAM2: Moondream2Analyzer,
        AnalyzerType.FLORENCE2: Florence2Analyzer,
        AnalyzerType.BLIP_BASE: BLIPBaseAnalyzer,
    }

    analyzer_class = analyzers.get(analyzer_type)
    if not analyzer_class:
        raise ValueError(f"Analyzer type non supportato: {analyzer_type}")

    return analyzer_class(data_path)


# ============================================================================
# CLI TEST
# ============================================================================

def test_single_image(image_path: str, analyzer_type: AnalyzerType = None):
    """Test rapido su singola immagine"""
    from PIL import Image

    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata: {image_path}")
        return

    # Se non specificato, testa tutti
    types_to_test = [analyzer_type] if analyzer_type else list(AnalyzerType)

    print(f"\n{'='*60}")
    print(f"TEST IMMAGINE: {image_path}")
    print(f"{'='*60}\n")

    results = []

    for atype in types_to_test:
        print(f"\n--- {atype.value.upper()} ---")
        try:
            analyzer = create_analyzer(atype)
            result = analyzer.analyze(image_path)
            results.append(result)

            print(f"Tempo: {result.inference_time:.2f}s")
            print(f"Caption base: {result.base_caption[:100]}...")
            print(f"Prompt finale: {result.prompt[:150]}...")
            print(f"Negative: {result.negative_prompt[:80]}...")

        except Exception as e:
            print(f"Errore: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Offline Image Analyzers")
    parser.add_argument("-i", "--image", required=True, help="Path immagine da analizzare")
    parser.add_argument("-m", "--model", choices=["moondream2", "florence2", "blip_base"],
                        help="Modello specifico (default: tutti)")

    args = parser.parse_args()

    analyzer_type = None
    if args.model:
        analyzer_type = AnalyzerType(args.model)

    test_single_image(args.image, analyzer_type)
