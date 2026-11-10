/**
 * QuPath 0.7 : exporte les annotations du projet ouvert en masques binaires, au format des masques PRMI.
 *
 * Projet attendu : data/interim/qupath/<lot>/projet_qupath/ (créé par import_masks.groovy).
 * Sortie : data/processed/masks_corriges/<lot>/GT_<nom de l'image>.png, un canal, 0 ou 255, même taille que l'image.
 * Toutes les annotations comptent comme racine, avec ou sans classe (une annotation dessinée sans choisir de classe
 * n'est pas perdue). Une image sans annotation donne un masque vide. Les masques déjà exportés sont remplacés.
 *
 * Usage : ouvrir le projet du lot, puis Automate > Script editor > Run (aucun réglage).
 */
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.RenderingHints
import javax.imageio.ImageIO

def project = getProject()
def projectDir = project.getPath().getParent().toFile()          // .../data/interim/qupath/<lot>/projet_qupath
def lot = projectDir.getParentFile().getName()
def dataDir = projectDir.getParentFile().getParentFile().getParentFile().getParentFile()   // .../data
def outDir = new File(dataDir, 'processed/masks_corriges/' + lot)
outDir.mkdirs()

int total = 0, unclassified = 0
project.getImageList().each { entry ->
    def imageData = entry.readImageData()
    def server = imageData.getServer()
    int w = server.getWidth(), h = server.getHeight()
    def anns = imageData.getHierarchy().getAnnotationObjects()
    unclassified += anns.count { it.getPathClass() == null }

    def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    def g = mask.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
    g.setColor(Color.BLACK); g.fillRect(0, 0, w, h)
    g.setColor(Color.WHITE)
    anns.each { g.fill(it.getROI().getShape()) }
    g.dispose()
    ImageIO.write(mask, 'png', new File(outDir, 'GT_' + entry.getImageName() + '.png'))
    total++
}
print("Terminé : " + total + " masques dans " + outDir + (unclassified ? " (dont " + unclassified + " annotations sans classe, comptées comme racine)" : ""))
