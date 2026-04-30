import random
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# ========== 全局配置 ==========
random.seed(42)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

MAX_MOL_WT = 1500
FILTER_ATOMS = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H'}

# ========== 基础过滤函数 ==========
def filter_basic(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False

        if Descriptors.MolWt(mol) >= MAX_MOL_WT:
            return False

        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
    except:
        return False
    return True


def filter_with_atoms(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False

        if Descriptors.MolWt(mol) >= MAX_MOL_WT:
            return False

        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
            if atom.GetSymbol() not in FILTER_ATOMS:
                return False
    except:
        return False
    return True


# ========= 结构标准化函数 ==========
def mol_to_features(mol):
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    inchi = Chem.MolToInchi(mol)
    inchikey = Chem.InchiToInchiKey(inchi)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    return canonical_smiles, inchi, inchikey, formula


# ========= 通用清洗模块 ==========
def clean_smiles_set(smiles_set, desc="Cleaning"):
    cleaned = []
    seen_inchi = set()

    for smi in tqdm(smiles_set, desc=desc):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            if not filter_with_atoms(mol):
                continue

            canonical_smiles, inchi, inchikey, formula = mol_to_features(mol)

            if inchi in seen_inchi:
                continue

            seen_inchi.add(inchi)

            cleaned.append({
                "canonical_smiles": canonical_smiles,
                "inchi": inchi,
                "inchikey": inchikey,
                "formula": formula
            })
        except:
            continue

    return cleaned


############################################
########## DSSTox DATASET MODULE ###########
############################################

print("\n[1/1] Cleaning DSSTox dataset...")

dsstox_raw_smiles = set()

base_path = r"D:\实验\数据\DSSTox_Feb_2024"

for i in range(1, 14):
    file_path = rf"{base_path}\DSSToxDump{i}.xlsx"
    print("正在读取：", file_path)

    try:
        df = pd.read_excel(file_path)

        if "SMILES" in df.columns:
            dsstox_raw_smiles.update(df["SMILES"].dropna())
    except Exception as e:
        print(f"⚠️ 读取失败: {file_path}，错误: {e}")
        continue

print(f"Raw DSSTox size: {len(dsstox_raw_smiles)}")

dsstox_cleaned = clean_smiles_set(
    dsstox_raw_smiles,
    desc="DSSTox cleaning"
)

dsstox_df = pd.DataFrame(dsstox_cleaned)

dsstox_df.to_csv(r"D:\实验\数据\DSSTox_Feb_2024\dsstox_cleaned.csv", index=False)

print(f"✅ DSSTox cleaned size: {len(dsstox_df)}")


############################################
############## TRAIN / VAL SPLIT ###########
############################################

print("\nSplitting train / validation...")

random.shuffle(dsstox_cleaned)

split_ratio = 0.9
cutoff = int(len(dsstox_cleaned) * split_ratio)

train_data = dsstox_cleaned[:cutoff]
val_data = dsstox_cleaned[cutoff:]

train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)

train_df.to_csv(r"D:\实验\数据\DSSTox_Feb_2024\train_cleaned.csv", index=False)
val_df.to_csv(r"D:\实验\数据\DSSTox_Feb_2024\val_cleaned.csv", index=False)

print(f"✅ Train size: {len(train_df)}")
print(f"✅ Val size: {len(val_df)}")
