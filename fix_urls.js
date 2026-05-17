const fs = require('fs');
const path = require('path');

function walk(dir) {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(function(file) {
        file = path.join(dir, file);
        const stat = fs.statSync(file);
        if (stat && stat.isDirectory()) {
            results = results.concat(walk(file));
        } else {
            if (file.endsWith('.tsx') || file.endsWith('.ts')) {
                results.push(file);
            }
        }
    });
    return results;
}

const files = walk('./app');

files.forEach(file => {
    let content = fs.readFileSync(file, 'utf8');
    let changed = false;
    
    // Pattern for double quotes: "http://localhost:5328/..."
    if (content.includes('"http://localhost:5328')) {
        content = content.replace(/"http:\/\/localhost:5328([^"]*)"/g, "`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5328'}$1`");
        changed = true;
    }
    
    // Pattern for single quotes: 'http://localhost:5328/...'
    if (content.includes("'http://localhost:5328")) {
        content = content.replace(/'http:\/\/localhost:5328([^']*)'/g, "`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5328'}$1`");
        changed = true;
    }
    
    // Pattern for template literals: `http://localhost:5328/...`
    if (content.includes("`http://localhost:5328")) {
        content = content.replace(/`http:\/\/localhost:5328([^`]*)`/g, "`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5328'}$1`");
        changed = true;
    }
    
    if (changed) {
        fs.writeFileSync(file, content, 'utf8');
        console.log('Updated: ' + file);
    }
});
