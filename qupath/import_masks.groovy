/**
 * QuPath 0.7 : crée le projet d'un lot (train, val ou test) et importe les masques d'origine en annotations.
 *
 * Entrée : data/interim/qupath/<lot>/, écrit par `prmi.qupath_io` (un sous-dossier par image avec image.jpg et
 * mask.png, masque sur un canal, 0 ou 255 ; lot.csv pour les métadonnées).
 * Sortie : projet QuPath dans data/interim/qupath/<lot>/projet_qupath/, une annotation `Racine` par morceau de racine
 * (composante connexe du masque), métadonnées groupe, type, racine, position, date et tube sur chaque image.
 *
 * Usage : régler lotDir ci-dessous, puis Automate > Script editor > Run (aucun projet n'a besoin d'être ouvert).
 */
import ij.IJ
import ij.process.ByteProcessor
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.common.ColorTools
import qupath.lib.roi.RoiTools
import qupath.imagej.tools.IJTools
import qupath.lib.regions.ImagePlane
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.images.ImageData
import qupath.lib.projects.Projects
import qupath.lib.gui.QuPathGUI
import java.awt.image.BufferedImage
import javafx.application.Platform

// ===================== CONFIG =====================
def lotDir = new File('/home/francois/Bureau/e-portfolio/prmi-roots/data/interim/qupath/val')   // train, val ou test
// ==================================================

def racine = PathClass.fromString('Racine', ColorTools.packRGB(30, 100, 255))   // bleu des superpositions du projet

// --- Images du lot (dossiers contenant image.jpg) ---
def imageDirs = []
lotDir.eachDir { d -> if (new File(d, 'image.jpg').exists()) imageDirs << d }
imageDirs.sort { it.name }
if (imageDirs.isEmpty()) { print("Aucune image.jpg trouvée dans " + lotDir); return }

// --- Métadonnées (lot.csv, écrit par prmi.qupath_io ; aucune virgule dans les valeurs) ---
def lines = new File(lotDir, 'lot.csv').readLines()
def header = lines[0].split(',', -1) as List
def meta = lines.drop(1).collectEntries { line ->
    def v = line.split(',', -1) as List
    def row = [header, v].transpose().collectEntries { it }
    [(row.name): row]
}
def keys = ['group': 'groupe', 'type': 'type', 'has_root': 'racine', 'position': 'position', 'date': 'date', 'tube_id': 'tube']

// --- Projet ---
def projectDir = new File(lotDir, 'projet_qupath')
if (projectDir.exists() && projectDir.listFiles()?.length > 0) {
    print("ATTENTION : " + projectDir + " existe déjà et n'est pas vide (annotations en cours ?). Rien n'est fait.")
    return
}
projectDir.mkdirs()
def project = Projects.createProject(projectDir, BufferedImage.class)
project.setPathClasses([racine])

// --- Masque -> une annotation par morceau de racine ---
def maskToAnnotations = { File maskFile ->
    def imp = IJ.openImage(maskFile.getPath())
    def cp = imp.getProcessor()
    int w = imp.getWidth(), h = imp.getHeight()
    def bp = new ByteProcessor(w, h)
    boolean any = false
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (cp.get(x, y) > 127) { bp.set(x, y, 255); any = true }
    if (!any) return []                                      // image sans racine : aucune annotation
    bp.setThreshold(127, 255, ij.process.ImageProcessor.NO_LUT_UPDATE)
    def ijRoi = new ij.plugin.filter.ThresholdToSelection().convert(bp)
    def roi = IJTools.convertToROI(ijRoi, 0.0, 0.0, 1.0, ImagePlane.getDefaultPlane())
    return RoiTools.splitROI(roi).collect { PathObjects.createAnnotationObject(it, racine) }
}

// --- Ajout des images et des annotations ---
int total = 0
imageDirs.each { dir ->
    def imgFile = new File(dir, 'image.jpg')
    def support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, imgFile.getCanonicalPath())
    def entry = project.addImage(support.builders.get(0))
    def imageData = entry.readImageData()
    imageData.setImageType(ImageData.ImageType.OTHER)

    def anns = maskToAnnotations(new File(dir, 'mask.png'))
    if (!anns.isEmpty()) { imageData.getHierarchy().addObjects(anns); total += anns.size() }

    entry.setImageName(dir.name)
    def row = meta[dir.name]
    if (row != null) keys.each { k, label -> entry.putMetadataValue(label, row[k]) }
    else print("Pas de métadonnées pour " + dir.name)
    entry.saveImageData(imageData)
}
project.syncChanges()

Platform.runLater { QuPathGUI.getInstance().setProject(project) }   // doit tourner sur le thread graphique
print("Terminé : " + imageDirs.size() + " images, " + total + " annotations (" + projectDir + ")")
