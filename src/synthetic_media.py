"""
Synthetic Multimodal Evidence Generator using Pillow (PIL).
Generates 5 realistic visual test cards in BOTH:
- `fotos_para_testar/` (with friendly Portuguese filenames so the user can drag-and-drop in Streamlit)
- `data/sample_media/` (for internal preset references)
"""

from __future__ import annotations
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_MEDIA_DIR = BASE_DIR / "data" / "sample_media"
USER_TEST_PHOTOS_DIR = BASE_DIR / "fotos_para_testar"


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_dispute_high_risk(path: Path) -> None:
    img = Image.new("RGB", (820, 520), color=(24, 26, 32))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, 820, 68], fill=(234, 29, 44))
    draw.text((24, 18), "iFood Trust & Safety — Evidência Fotográfica #IFD-98412", fill=(255, 255, 255), font=_get_font(22))

    draw.rectangle([24, 90, 380, 490], fill=(248, 246, 236), outline=(200, 195, 180), width=2)
    receipt_lines = [
        "CUPOM FISCAL / COMPROVANTE iFOOD",
        "Restaurante: Fogo & Brasa Prime",
        "Pedido: #IFD-98412 | Data: 15/05/2025",
        "----------------------------------------",
        "1x Combo Picanha Nobre 600g    R$ 139,90",
        "1x Combinado Sushi 24 pcs      R$  50,00",
        "----------------------------------------",
        "TOTAL PAGO (PIX):              R$ 189,90",
        "Tempo de Entrega: 31 min (No Prazo)",
        "Lacre de Segurança: #LC-9941-OK",
        "----------------------------------------",
        "Reclamação do Usuário:",
        "'Embalagem violada e item faltante'",
        "Histórico 90d: 14 reembolsos / 22 pedidos",
    ]
    y = 108
    for idx, line in enumerate(receipt_lines):
        color = (180, 20, 30) if "Reclamação" in line or "Histórico" in line else (30, 30, 30)
        draw.text((40, y), line, fill=color, font=_get_font(14 if idx > 0 else 15))
        y += 26

    draw.rectangle([405, 90, 796, 490], fill=(36, 40, 48), outline=(234, 29, 44), width=2)
    draw.text((425, 108), "INSPEÇÃO VISUAL (VISION AI):", fill=(255, 210, 90), font=_get_font(17))

    draw.rectangle([465, 155, 735, 365], fill=(198, 156, 109), outline=(140, 100, 60), width=3)
    draw.rectangle([535, 145, 665, 185], fill=(46, 160, 67), outline=(255, 255, 255), width=2)
    draw.text((548, 157), "LACRE INTACTO", fill=(255, 255, 255), font=_get_font(14))
    draw.text((495, 245), "EMBALAGEM TÉRMICA PREMIUM", fill=(50, 35, 20), font=_get_font(15))
    draw.text((515, 275), "Sem sinais de vazamento", fill=(60, 45, 25), font=_get_font(14))

    draw.rectangle([425, 385, 775, 472], fill=(60, 22, 28), outline=(234, 29, 44), width=1)
    draw.text((438, 396), "Score Integridade Embalagem: 92.0 / 100 (ÍNTEGRA)", fill=(255, 140, 140), font=_get_font(14))
    draw.text((438, 420), "Anomalia: Foto contradiz alegação de avaria!", fill=(255, 200, 120), font=_get_font(14))
    draw.text((438, 444), "Veredito Preliminar Vision: SUSPEITA DE FRAUDE", fill=(255, 90, 90), font=_get_font(14))

    img.save(str(path), format="PNG")


def _draw_dispute_legit_damaged(path: Path) -> None:
    img = Image.new("RGB", (820, 520), color=(22, 28, 34))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, 820, 68], fill=(22, 130, 95))
    draw.text((24, 18), "iFood Care — Evidência de Avaria Severa #IFD-77104", fill=(255, 255, 255), font=_get_font(22))

    draw.rectangle([24, 90, 380, 490], fill=(248, 246, 236), outline=(180, 185, 190), width=2)
    receipt_lines = [
        "COMPROVANTE DE PEDIDO iFOOD",
        "Restaurante: Nonna Mia Trattoria",
        "Pedido: #IFD-77104 | Data: 15/05/2025",
        "----------------------------------------",
        "1x Parmegiana Artesanal Gratin   R$ 54,90",
        "1x Suco Natural Laranja 500ml    R$ 13,60",
        "----------------------------------------",
        "TOTAL PAGO (Cartão):             R$ 68,50",
        "Tempo de Entrega: 78 min (ATRASO SEVERO)",
        "Lacre de Segurança: ROMPIDO NO TRANSPORTE",
        "----------------------------------------",
        "Reclamação do Cliente:",
        "'Embalagem esmagada, molho vazou na sacola'",
        "Histórico 90d: 1 reembolso / 38 pedidos (2.6%)",
    ]
    y = 108
    for idx, line in enumerate(receipt_lines):
        color = (15, 110, 75) if "Histórico" in line else (30, 30, 30)
        draw.text((40, y), line, fill=color, font=_get_font(14 if idx > 0 else 15))
        y += 26

    draw.rectangle([405, 90, 796, 490], fill=(38, 42, 48), outline=(46, 160, 67), width=2)
    draw.text((425, 108), "INSPEÇÃO VISUAL (VISION AI):", fill=(130, 240, 175), font=_get_font(17))

    draw.polygon([(470, 175), (720, 195), (745, 345), (455, 360)], fill=(175, 130, 90), outline=(220, 50, 50))
    draw.ellipse([510, 235, 695, 355], fill=(195, 35, 30))
    draw.text((525, 275), "VAZAMENTO DE MOLHO", fill=(255, 255, 255), font=_get_font(15))
    draw.rectangle([530, 150, 675, 188], fill=(210, 45, 45))
    draw.text((545, 160), "LACRE VIOLADO", fill=(255, 255, 255), font=_get_font(14))

    draw.rectangle([425, 385, 775, 472], fill=(20, 55, 42), outline=(46, 160, 67), width=1)
    draw.text((438, 396), "Score Integridade Embalagem: 18.5 / 100 (DANIFICADA)", fill=(150, 255, 190), font=_get_font(14))
    draw.text((438, 420), "Dano físico e vazamento confirmados por Visão.", fill=(220, 245, 230), font=_get_font(14))
    draw.text((438, 444), "Veredito Preliminar Vision: REEMBOLSO LEGÍTIMO", fill=(110, 255, 160), font=_get_font(14))

    img.save(str(path), format="PNG")


def _draw_menu_physical_messy(path: Path) -> None:
    img = Image.new("RGB", (820, 540), color=(244, 239, 226))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, 820, 72], fill=(45, 32, 26))
    draw.text((24, 20), "CARDÁPIO FÍSICO DIGITALIZADO — Burger Lab Artesanal", fill=(255, 215, 130), font=_get_font(21))

    draw.rectangle([24, 90, 796, 515], fill=(252, 250, 244), outline=(140, 115, 90), width=2)
    draw.text(
        (42, 106),
        "Itens extraídos via OCR (Contém descrições fracas de SEO e erro grave de precificação):",
        fill=(120, 50, 30),
        font=_get_font(15),
    )

    dishes = [
        ("1. Smash Duplo Casa", "pao carne e queijo", "R$ 36,90", False),
        ("2. Burger Trufado Real", "hamburguer com maionese", "R$ 44,90", False),
        ("3. Combo X-Salada Clássico", "lanche simples com refri", "R$ 450,00  [ANOMALIA CRÍTICA DE PREÇO!]", True),
        ("4. Batata Rústica Alecrim", "batata frita porcao", "R$ 24,90", False),
        ("5. Chicken Crispy Burger", "frango empanado no pao", "R$ 38,00", False),
        ("6. Milkshake Belga 400ml", "milkshake chocolate", "R$ 26,90", False),
    ]

    y = 148
    for title, desc, price_str, is_anom in dishes:
        box_fill = (255, 228, 228) if is_anom else (242, 238, 230)
        box_outline = (220, 35, 35) if is_anom else (200, 190, 175)
        draw.rectangle([42, y, 778, y + 52], fill=box_fill, outline=box_outline, width=2 if is_anom else 1)
        draw.text((56, y + 8), f"{title}  —  {price_str}", fill=(185, 15, 20) if is_anom else (35, 30, 25), font=_get_font(15))
        draw.text((56, y + 29), f"Descrição atual: '{desc}' (SEO Fraco)", fill=(100, 95, 90), font=_get_font(13))
        y += 60

    img.save(str(path), format="PNG")


def _draw_sushi_fraud_dark_photo(path: Path) -> None:
    img = Image.new("RGB", (820, 520), color=(30, 18, 24))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, 820, 68], fill=(185, 28, 28))
    draw.text((24, 18), "iFood Trust & Safety — Evidência Suspeita #IFD-99201 (Sushi Prime)", fill=(255, 255, 255), font=_get_font(21))

    draw.rectangle([24, 90, 380, 490], fill=(250, 245, 240), outline=(220, 180, 180), width=2)
    lines = [
        "COMPROVANTE PEDIDO #IFD-99201",
        "Restaurante: Sushi Leblon Express",
        "----------------------------------------",
        "1x Combinado Imperial 48 Peças R$ 219,00",
        "2x Temaki Salmão Trufado       R$  49,90",
        "----------------------------------------",
        "TOTAL PAGO (PIX):              R$ 268,90",
        "Entrega confirmada por PIN: SIM",
        "----------------------------------------",
        "Reclamação: 'Não recebi metade das peças'",
        "Alerta Conta: 9 estornos nos últimos 30d",
    ]
    y = 110
    for line in lines:
        draw.text((40, y), line, fill=(170, 20, 20) if "Alerta" in line else (30, 30, 30), font=_get_font(14))
        y += 28

    draw.rectangle([405, 90, 796, 490], fill=(45, 32, 38), outline=(234, 29, 44), width=2)
    draw.text((425, 110), "FOTO ENVIADA PELO USUÁRIO:", fill=(255, 180, 180), font=_get_font(17))
    draw.rectangle([455, 155, 745, 355], fill=(60, 65, 75), outline=(200, 200, 200), width=2)
    draw.text((490, 225), "BANDEJA LACRADA COM 48 PEÇAS", fill=(255, 255, 255), font=_get_font(15))
    draw.text((510, 255), "(Contagem Visual Vision AI: OK)", fill=(160, 240, 180), font=_get_font(14))
    draw.text((435, 395), "Veredito Vision AI: Tentativa de Golpe (Foto Intacta)", fill=(255, 110, 110), font=_get_font(15))

    img.save(str(path), format="PNG")


def _draw_pizza_menu_weak_seo(path: Path) -> None:
    img = Image.new("RGB", (820, 540), color=(248, 244, 232))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, 820, 72], fill=(153, 27, 27))
    draw.text((24, 20), "CARDÁPIO ENVIADO — Pizzaria Forno D'Oro (Auditoria MenuOps)", fill=(255, 255, 255), font=_get_font(21))

    dishes = [
        ("1. Pizza Calabresa", "pizza com calabresa e cebola", "R$ 59,90", False),
        ("2. Pizza Marguerita", "queijo tomate manjericao", "R$ 64,90", False),
        ("3. Pizza 4 Queijos Especial", "pizza de queijo", "R$ 690,00  [ERRO DE DIGITAÇÃO R$ 690!]", True),
        ("4. Brotinho Chocolate", "pizza doce", "R$ 34,90", False),
    ]
    y = 115
    for title, desc, price_str, is_anom in dishes:
        fill_c = (254, 226, 226) if is_anom else (255, 255, 255)
        draw.rectangle([42, y, 778, y + 75], fill=fill_c, outline=(220, 38, 38) if is_anom else (200, 190, 180), width=2)
        draw.text((58, y + 12), f"{title}  —  {price_str}", fill=(185, 28, 28) if is_anom else (25, 25, 25), font=_get_font(16))
        draw.text((58, y + 42), f"Texto atual: '{desc}' -> Precisa de Otimização IA", fill=(100, 90, 85), font=_get_font(14))
        y += 92

    img.save(str(path), format="PNG")


def ensure_sample_images() -> dict[str, str]:
    SAMPLE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    USER_TEST_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "dispute_high_risk": SAMPLE_MEDIA_DIR / "dispute_high_risk.png",
        "dispute_legit_damaged": SAMPLE_MEDIA_DIR / "dispute_legit_damaged.png",
        "menu_physical_messy": SAMPLE_MEDIA_DIR / "menu_physical_messy.png",
    }

    if not paths["dispute_high_risk"].exists():
        _draw_dispute_high_risk(paths["dispute_high_risk"])
    if not paths["dispute_legit_damaged"].exists():
        _draw_dispute_legit_damaged(paths["dispute_legit_damaged"])
    if not paths["menu_physical_messy"].exists():
        _draw_menu_physical_messy(paths["menu_physical_messy"])

    friendly_map = {
        "1_golpe_embalagem_intacta_churrasco.png": paths["dispute_high_risk"],
        "2_cliente_honesto_embalagem_rasgada_molho_vazado.png": paths["dispute_legit_damaged"],
        "3_cardapio_hamburgueria_preco_errado_450_reais.png": paths["menu_physical_messy"],
    }
    for fname, src_path in friendly_map.items():
        dst = USER_TEST_PHOTOS_DIR / fname
        if not dst.exists():
            shutil.copyfile(src_path, dst)

    extra_4 = USER_TEST_PHOTOS_DIR / "4_golpe_sushi_268_reais_lacre_intacto.png"
    if not extra_4.exists():
        _draw_sushi_fraud_dark_photo(extra_4)

    extra_5 = USER_TEST_PHOTOS_DIR / "5_cardapio_pizzaria_erro_690_reais.png"
    if not extra_5.exists():
        _draw_pizza_menu_weak_seo(extra_5)

    return {k: str(v) for k, v in paths.items()}


def get_user_test_photos() -> dict[str, str]:
    ensure_sample_images()
    return {p.name: str(p) for p in sorted(USER_TEST_PHOTOS_DIR.glob("*.png"))}
