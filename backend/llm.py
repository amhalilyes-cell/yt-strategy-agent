import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_INSIGHTS = """Tu es un stratège contenu YouTube orienté business.
Tu produis des insights actionnables, adaptés au marché FR, orientés objectif
(visibilité, leads, vente, autorité, lancement).
Pas de blabla. Que du concret.
"""

SYSTEM_PACK = """Tu es un agent IA de stratégie contenu.
Tu dois produire un "pack" prêt à publier: hooks, idées, titres, scripts shorts, script long, plan 30 jours.
Tout doit être orienté objectif business, concret, structuré, et adapté marché FR.
"""

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquante dans .env")
    return OpenAI(api_key=api_key)

def make_insights(niche: str, objectif: str, niveau: str, freq: int, videos_dump: str) -> str:
    client = _client()
    prompt = f"""Contexte:
- Niche: {niche}
- Objectif: {objectif}
- Niveau: {niveau}
- Fréquence: {freq}/semaine

Analyse ces vidéos (US) et adapte en stratégie FR.
Donne :
1) Formats gagnants
2) Ce qui est saturé
3) Ce qui est sous-exploité
4) 7 recommandations stratégiques
5) 5 angles à éviter
6) 5 angles opportunité

Données:
{videos_dump}
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_INSIGHTS},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return resp.choices[0].message.content or ""

def make_pack(niche: str, objectif: str, niveau: str, freq: int, insights: str):
    client = _client()

    schema = {
        "type": "object",
        "properties": {
            "hooks": {
                "type": "object",
                "properties": {
                    "curiosite": {"type": "array", "items": {"type": "string"}},
                    "preuve": {"type": "array", "items": {"type": "string"}},
                    "polarisation": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["curiosite", "preuve", "polarisation"],
            },
            "ideas": {"type": "array", "items": {"type": "string"}},
            "titles": {"type": "array", "items": {"type": "string"}},
            "scripts_shorts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hook": {"type": "string"},
                        "body": {"type": "array", "items": {"type": "string"}},
                        "cta": {"type": "string"},
                    },
                    "required": ["hook", "body", "cta"],
                },
            },
            "script_long": {
                "type": "object",
                "properties": {
                    "hook": {"type": "string"},
                    "probleme": {"type": "string"},
                    "cadre": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"}},
                    "objections": {"type": "array", "items": {"type": "string"}},
                    "conclusion": {"type": "string"},
                    "cta": {"type": "string"},
                },
                "required": ["hook", "probleme", "cadre", "points", "objections", "conclusion", "cta"],
            },
            "plan_30_jours": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "jour": {"type": "integer"},
                        "format": {"type": "string"},
                        "theme": {"type": "string"},
                        "objectif": {"type": "string"},
                        "cta": {"type": "string"},
                    },
                    "required": ["jour", "format", "theme", "objectif", "cta"],
                },
            },
        },
        "required": ["hooks", "ideas", "titles", "scripts_shorts", "script_long", "plan_30_jours"],
    }

    prompt = f"""Contexte:
- Niche: {niche}
- Objectif: {objectif}
- Niveau: {niveau}
- Fréquence: {freq}/semaine

Insights (à respecter):
{insights}

Contraintes:
- Marché FR (langage FR, exemples FR)
- Orienté business (CTA lead magnet / call / DM / audit)
- Hooks très agressifs mais crédibles
- Idées et titres variés (pas de répétitions)

Rends UNIQUEMENT du JSON valide selon le schéma.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PACK},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        response_format={"type": "json_schema", "json_schema": {"name": "pack", "schema": schema}},
    )

    txt = resp.choices[0].message.content or "{}"
    return json.loads(txt)