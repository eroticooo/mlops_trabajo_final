# archivo: llamar_api_prediccion.py
import json
import requests
from typing import Any, Dict, Tuple

BASE_URL = "https://mlops-copper-688b09eaaf1b.herokuapp.com"
OPENAPI_URL = f"{BASE_URL}/openapi.json"

def fetch_openapi() -> Dict[str, Any]:
    r = requests.get(OPENAPI_URL, timeout=20)
    r.raise_for_status()
    return r.json()

def deref(schema: Dict[str, Any], ref: str) -> Dict[str, Any]:
    assert ref.startswith("#/"), "Solo refs locales soportadas"
    target = schema
    for part in ref[2:].split("/"):
        target = target[part]
    return target

def get_post_endpoint(spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    paths = spec.get("paths", {})
    if "/prediccion/" in paths and "post" in paths["/prediccion/"]:
        return "/prediccion/", paths["/prediccion/"]["post"]
    for p, item in paths.items():
        if "post" in item:
            return p, item["post"]
    raise RuntimeError("No encontré endpoints POST en el OpenAPI.")

def extract_body_schema(spec: Dict[str, Any], post_item: Dict[str, Any]) -> Dict[str, Any]:
    rb = post_item.get("requestBody", {})
    content = rb.get("content", {})
    app_json = content.get("application/json", {})
    schema = app_json.get("schema", {})
    if "$ref" in schema:
        return deref(spec, schema["$ref"])
    return schema

def get_properties_and_required(schema: Dict[str, Any], spec: Dict[str, Any]):
    if "$ref" in schema:
        schema = deref(spec, schema["$ref"])
    if schema.get("type") == "array" and "items" in schema:
        inner = schema["items"]
        if "$ref" in inner:
            inner = deref(spec, inner["$ref"])
        schema = inner
    if "allOf" in schema:
        merged = {"properties": {}, "required": []}
        for part in schema["allOf"]:
            if "$ref" in part:
                part = deref(spec, part["$ref"])
            merged["properties"].update(part.get("properties", {}))
            merged["required"] += part.get("required", [])
        schema = merged
    props = schema.get("properties", {})
    req = set(schema.get("required", []))
    return props, req

def guess_default(prop_schema: Dict[str, Any]):
    if "example" in prop_schema:
        return prop_schema["example"]
    if "default" in prop_schema:
        return prop_schema["default"]
    typ = prop_schema.get("type")
    if typ == "integer":
        return 0
    if typ == "number":
        return 0.0
    if typ == "boolean":
        return False
    if typ == "array":
        return []
    return ""

def cast_input(user_str: str, typ: str):
    if user_str == "":
        return None
    if typ == "integer":
        return int(user_str)
    if typ == "number":
        return float(user_str)
    if typ == "boolean":
        return user_str.strip().lower() in {"true", "1", "yes", "si", "sí"}
    return user_str

def main():
    print(f"🔎 Descargando OpenAPI de: {OPENAPI_URL}")
    spec = fetch_openapi()

    path, post_item = get_post_endpoint(spec)
    print(f"✅ Endpoint POST detectado: {path}")

    schema = extract_body_schema(spec, post_item)
    props, required = get_properties_and_required(schema, spec)

    if not props:
        print("⚠️ No pude encontrar propiedades en el body. Enviaré {} como payload.")
        payload = {}
    else:
        print("\n📋 Campos detectados (req = requerido):\n")
        for name, psch in props.items():
            typ = psch.get("type", "string")
            print(f" - {name}  (tipo: {typ}, req: {'sí' if name in required else 'no'})")

        print("\n✍️ Ingresa valores (Enter para aceptar sugerencia).")
        payload = {}
        for name, psch in props.items():
            typ = psch.get("type", "string")
            suggested = guess_default(psch)
            shown = json.dumps(suggested) if not isinstance(suggested, (int, float, str, bool)) else suggested
            user_val = input(f"{name} [{typ}] (sugerido={shown}): ").strip()
            if user_val == "":
                payload[name] = suggested
            else:
                try:
                    casted = cast_input(user_val, typ)
                    if casted is None:
                        casted = suggested
                    payload[name] = casted
                except Exception as e:
                    print(f"   ↳ Valor inválido, uso sugerido ({shown}). Detalle: {e}")
                    payload[name] = suggested

    # ✅ Normalización rápida: reemplaza '_' por ' ' en llaves (por si el modelo se entrenó con espacios)
    payload_norm = {k.replace("_", " "): v for k, v in payload.items()}

    url = f"{BASE_URL}{path}"
    print("\n🔧 Payload (normalizado con espacios en llaves):")
    print(json.dumps(payload_norm, indent=2, ensure_ascii=False))

    print("\n🚀 Enviando POST a", url)
    r = requests.post(url, json=payload_norm, timeout=30)

    print(f"\n📨 Status: {r.status_code}")
    try:
        print("🧾 Respuesta JSON:")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print("🧾 Respuesta texto:")
        print(r.text)

if __name__ == "__main__":
    main()
