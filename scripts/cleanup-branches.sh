#!/bin/bash
# Script pour gérer la visibilité des branches
# Usage: ./cleanup-branches.sh [--dry-run]

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Mode dry-run par défaut
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}Mode DRY-RUN activé - Aucune modification ne sera effectuée${NC}"
    echo ""
fi

# Fonction pour afficher un titre
print_header() {
    echo ""
    echo -e "${GREEN}=== $1 ===${NC}"
    echo ""
}

# Fonction pour afficher un avertissement
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Fonction pour afficher une erreur
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Fonction pour afficher un succès
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_header "Vérification des branches"

# Mettre à jour les références distantes
echo "Mise à jour des références distantes..."
git fetch --prune

# Lister toutes les branches distantes
echo ""
echo "Branches distantes actuelles :"
git branch -r | grep -v HEAD

print_header "Analyse des branches à supprimer"

# Branches à supprimer (non-main branches)
BRANCHES_TO_DELETE=()

# Parcourir toutes les branches distantes
while IFS= read -r branch; do
    # Nettoyer le nom de la branche
    branch_name=$(echo "$branch" | sed 's/origin\///' | xargs)
    
    # Ignorer HEAD et main
    if [[ "$branch_name" == "HEAD" ]] || [[ "$branch_name" == "main" ]]; then
        continue
    fi
    
    BRANCHES_TO_DELETE+=("$branch_name")
done < <(git branch -r | grep -v HEAD)

# Afficher les branches à supprimer
if [ ${#BRANCHES_TO_DELETE[@]} -eq 0 ]; then
    print_success "Aucune branche à supprimer. Seule la branche 'main' existe."
    exit 0
fi

echo "Branches qui seront supprimées :"
for branch in "${BRANCHES_TO_DELETE[@]}"; do
    print_warning "  - $branch"
done

echo ""
echo "Total : ${#BRANCHES_TO_DELETE[@]} branche(s)"

# Demander confirmation si pas en mode dry-run
if [ "$DRY_RUN" = false ]; then
    echo ""
    read -p "Voulez-vous supprimer ces branches ? (oui/yes/y) : " -r
    echo ""
    
    if [[ ! $REPLY =~ ^([Oo]ui|[Yy]es|[Yy])$ ]]; then
        print_warning "Opération annulée par l'utilisateur"
        exit 0
    fi
fi

print_header "Suppression des branches"

# Supprimer les branches
for branch in "${BRANCHES_TO_DELETE[@]}"; do
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Supprimerait la branche : $branch"
    else
        echo "Suppression de : $branch"
        if git push origin --delete "$branch" 2>/dev/null; then
            print_success "Branche '$branch' supprimée"
        else
            print_error "Échec de la suppression de '$branch'"
        fi
    fi
done

# Nettoyer les références locales
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Nettoyage des références locales..."
    git fetch --prune
fi

print_header "Résumé"

if [ "$DRY_RUN" = true ]; then
    echo "Mode DRY-RUN : Aucune modification effectuée"
    echo "Relancez sans --dry-run pour effectuer les suppressions"
else
    print_success "Nettoyage terminé !"
fi

echo ""
echo "Vérification finale des branches :"
git branch -r | grep -v HEAD

echo ""
print_success "Seule la branche 'main' devrait être visible maintenant"
